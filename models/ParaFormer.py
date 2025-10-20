import torch
from torch import nn 
from torch.nn.parallel import parallel_apply
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time


############################### Used in Single GPU training and testing ##############################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * heads, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) + x

class ParaFormer(nn.Module):

    class Branches(nn.Module):
        def __init__(self, dim_X, num_branches, branch_depth, num_heads, dim_head, dim_MLP, dropout=0., available_devices=[]):
            super().__init__()
            self.num_branches = num_branches
            self.available_devices=available_devices
            self.device_map=[]

            if not self.available_devices: # for single GPUs
                self.branches = nn.ModuleList([
                    self._make_branch(dim_X, branch_depth, num_heads, dim_head, dim_MLP, dropout)
                    for _ in range(num_branches)
                ])
            else: # for mutiple GPUs
                self.device_map = [self.available_devices[i % len(self.available_devices)] for i in range(num_branches)]
                self.branches = nn.ModuleList([
                    self._make_branch(dim_X, branch_depth, num_heads, dim_head, dim_MLP, dropout, device=self.device_map[i])
                    for i in range(num_branches)
                ])

        def _make_branch(self, dim_X, branch_depth, num_heads, dim_head, dim_MLP, dropout, device=None):
            layers = []
            for _ in range(branch_depth):
                layers.extend([
                    PreNorm(dim_X, Attention(dim_X, num_heads, dim_head, dropout)),
                    PreNorm(dim_X, FeedForward(dim_X, dim_MLP, dropout))])
            return nn.Sequential(*layers) if device==None else nn.Sequential(*layers).to(device)

        def forward(self, x):   
            if self.device_map: # for multiple GPUs
                inputs = [x.to(f'cuda:{self.device_map[i]}') for i in range(self.num_branches)]
                outputs = parallel_apply(self.branches, inputs, devices = self.device_map)
                outputs = [out.to(self.device_map[0]) for out in outputs]
                return torch.sum(torch.stack(outputs), dim=0)
            else: # for single GPU
                outputs = []
                for branch in self.branches:
                    outputs.append(branch(x))
                return torch.sum(torch.stack(outputs), dim=0)

    def __init__(self, *, input_image_size, patch_size, dim_X, num_branches, branch_depth, num_heads, dim_MLP, num_classes_to_predict, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., label=None, available_devices=[]):
        super().__init__()

        self.available_devices=available_devices
        self.num_branches = num_branches
        self.branch_depth = branch_depth
        
        ### embedding extractor
        image_height, image_width = pair(input_image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim_X),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_X, device=available_devices[0] if available_devices else None))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_X, device=available_devices[0] if available_devices else None))
        self.dropout = nn.Dropout(emb_dropout)

        ### braches creation
        self.paraformerbranches = ParaFormer.Branches(dim_X, num_branches, branch_depth, num_heads, dim_head, dim_MLP, dropout, available_devices)
      
        ### pooling after the branches
        self.pool = pool
        self.to_latent = nn.Identity()

        ### MLP layers
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_X),
            nn.Linear(dim_X, num_classes_to_predict)
        )

        self.main_device = None
        if available_devices: # for multiple GPU
            print('multiple GPU version initated ....')
            self.main_device = available_devices[0]
            self.to_patch_embedding = self.to_patch_embedding.to(self.main_device)
            self.dropout = self.dropout.to(self.main_device)
            self.mlp_head = self.mlp_head.to(self.main_device)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.paraformerbranches(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    
    # used in train and test depth = n, parallel = m
    def forward_step(self, img, num_branches, feature_flag = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if self.available_devices: # for multiple GPUs
            self.inputs = [x.to(self.paraformerbranches.device_map[i]) for i in range(num_branches+1)]
            branches_to_run = self.paraformerbranches.branches[:num_branches+1]
            outputs = parallel_apply(branches_to_run, self.inputs[:num_branches+1], devices = self.paraformerbranches.device_map[:num_branches+1])
            x = torch.sum(torch.stack([out.to(self.available_devices[0]) for out in outputs]), dim=0)
        else: # for single GPU
            outputs = []
            for i in range(num_branches+1):
                outputs.append(self.paraformerbranches.branches[i](x))

            x = torch.sum(torch.stack(outputs), dim=0)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        feature = x
        x = self.to_latent(x)
        if feature_flag:
            return self.mlp_head(x), feature
        else:
            return self.mlp_head(x)

    # used in test: depth = n, parallel = 1 
    def forward_to_depth(self, img, num_branches, branch_depth, feature_flag=False):

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        for block_idx in range(num_branches + 1):
            if block_idx < num_branches:
                x = self.transformer.blocks[block_idx](x)
            else:
                block = self.transformer.blocks[block_idx]
                for layer_idx, layer in enumerate(block):
                    if layer_idx // 2 > branch_depth: 
                        break
                    x = layer(x)
    
        x = x[:, 0]
        feature = x
        x = self.to_latent(x)
        
        if feature_flag:
            return self.mlp_head(x), feature
        else:
            return self.mlp_head(x)
