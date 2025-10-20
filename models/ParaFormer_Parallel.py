import torch
from torch import nn 
from torch.nn.parallel import parallel_apply
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



############################### Used in Multiple GPU training and testing ##############################

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


class Paraformer(nn.Module):
    def __init__(self, dim, num_blocks, heads, dim_head, mlp_dim, depth, dropout=0.):
        super().__init__()
        self.num_blocks = num_blocks
        self.available_devices = list(range(torch.cuda.device_count()))
        self.device_map = [self.available_devices[i % len(self.available_devices)] for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            self._make_block(dim, heads, dim_head, mlp_dim, depth, dropout, device=self.device_map[i])
            for i in range(num_blocks)
        ])

    def _make_block(self, dim, heads, dim_head, mlp_dim, depth, dropout, device):
        layers = []
        for _ in range(depth):
            layers.extend([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))])
        return nn.Sequential(*layers).to(device)

    def forward(self, x):
        inputs = [x.to(f'cuda:{self.device_map[i]}') for i in range(self.num_blocks)]
        outputs = parallel_apply(self.blocks, inputs, devices = self.device_map)
        outputs = [out.to(self.device_map[0]) for out in outputs]
        return torch.sum(torch.stack(outputs), dim=0)


class ViP(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, num_blocks, heads, mlp_dim, depth, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., label=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.depth = depth
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim, device='cuda:0'))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim, device='cuda:0'))
        self.dropout = nn.Dropout(emb_dropout)

        self.paraformer = Paraformer(dim, num_blocks, heads, dim_head, mlp_dim, depth, dropout)
      
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.main_device = 'cuda:0'
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
        x = self.paraformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    

    def forward_step(self, img, layer_number, feature_flag = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        self.inputs = [x.to(self.paraformer.device_map[i]) for i in range(layer_number+1)]
        blocks_to_run = self.paraformer.blocks[:layer_number+1]
        outputs = parallel_apply(blocks_to_run, self.inputs[:layer_number+1], devices = self.paraformer.device_map[:layer_number+1])
        x = torch.sum(torch.stack([out.to('cuda:0') for out in outputs]), dim=0)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        feature = x
        x = self.to_latent(x)
        if feature_flag:
            return self.mlp_head(x), feature
        else:
            return self.mlp_head(x)