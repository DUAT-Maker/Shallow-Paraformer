# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '4' 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar

from models.ParaFormer2 import ParaFormer
from collections import OrderedDict
from data_utils import get_loader


##### Training
def train(args, net, epoch, trainloader, criterion, scaler, optimizer, use_amp, device = 'cuda:0'):
    print('\nEpoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for i in range(args.num_branches):
            with torch.amp.autocast(enabled=use_amp,device_type='cuda'):
                outputs = net.forward_step(inputs, i)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            _, predicted = outputs.max(1)
            batch_num = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx*args.num_branches+i, len(trainloader)*args.num_branches, 'LPA Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss, 100.*correct_batch/batch_num, correct_batch, batch_num))

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast(enabled=use_amp,device_type='cuda'):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), 100.*correct/total

def val(args, net, epoch, dataloader, criterion, best_acc, device = 'cuda:0'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        os.makedirs(f'checkpoint/{args.dataset}', exist_ok=True)
        torch.save(net.state_dict(), f'checkpoint/{args.dataset}/{args.net}_depth_{args.branch_depth}_parallel_{args.num_branches}_epoch_{args.n_epochs}.t7')
        best_acc = acc

    content = time.ctime() + ' ' + f'Val Epoch {epoch}, val loss: {test_loss/(batch_idx+1):.5f}, acc: {(acc):.5f}'
    print(content)
   
    return test_loss/(batch_idx+1), acc

def test(net, epoch, dataloader, criterion, device = 'cuda:0'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    epoch_loss = test_loss / (batch_idx+1)
    epoch_acc = 100. * correct / total
    
    content = time.ctime() + ' ' + f'Test Epoch {epoch}, Test loss: {epoch_loss:.5f}, acc: {epoch_acc:.5f}'
    print(content)
    return epoch_loss, epoch_acc


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch ViT Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--pretrained_dir', default=None, type=str,help='resume from checkpoint')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--aug', action='store_false', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--dp', default=True, type=bool,help='use data parallel')
    parser.add_argument('--bs', default=500)
    parser.add_argument('--multi_GPUs', default=False, type=bool, help='use multiple GPUs')
    parser.add_argument('--branch_depth', default=2, type=int) 
    parser.add_argument('--num_branches', default=8, type=int)
                        
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--dim_X', default=192, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--dim_MLP', default=768, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--emb_dropout', default=0.1, type=float)
    parser.add_argument('--data_dir', default='/home/lookwei/datasets', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100','FashionMNIST'])
 
    args = parser.parse_args()
    classes_list = {"cifar10": 10, "cifar100": 100, 'FashionMNIST': 10}
    size_list = {"cifar10": 224, "cifar100": 224, 'FashionMNIST': 224}
    patch_size_list = {"cifar10": 16, "cifar100": 16, 'FashionMNIST': 16}
    img_C_list = {"cifar10": 3, "cifar100": 3, 'FashionMNIST': 1}

    num_classes = classes_list[args.dataset]
    args.size = size_list[args.dataset]
    args.patch = patch_size_list[args.dataset]
    args.img_C = img_C_list[args.dataset]

    use_amp = not args.noamp
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, valloader, testloader = get_loader(args)

    print('==> Building model..')
    net = ParaFormer(
    input_image_size = args.size,
    patch_size = args.patch,
    num_classes_to_predict = num_classes,
    dim_X = args.dim_X,
    num_branches = args.num_branches,
    num_heads = args.num_heads,
    dim_MLP = args.dim_MLP,
    branch_depth = args.branch_depth,
    dropout = args.dropout,
    emb_dropout = args.emb_dropout,
    channels = args.img_C,
    available_devices = list(range(torch.cuda.device_count())) if args.multi_GPUs else []
    )

    print(net)

    criterion = nn.CrossEntropyLoss()
    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr) 
        
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #net.cuda()

    if device=='cuda':
        if args.multi_GPUs:
            available_devices = list(range(torch.cuda.device_count()))
            device="cuda:0"
        else:   
            net.cuda()


    num_params = count_parameters(net)
    print("Total Parameter: \t%2.1fM" % num_params)

    history = {
    'epoch_elapse': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'test_loss': [],
    'test_acc': [],
    }

    os.makedirs("vit_train_logs_para", exist_ok=True)

    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()
        train_loss, train_acc = train(args, net, epoch, trainloader, criterion, scaler, optimizer, use_amp, device)
        end_time = time.time()

        val_loss, val_acc = val(args, net, epoch, valloader, criterion, best_acc, device)
        test_loss, test_acc = test(net, epoch, testloader, criterion, device)

        history['epoch_elapse'].append(end_time-start_time)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    torch.save(history, f'vit_train_logs_para/dataset_{args.dataset}_model_{args.net}_epoch_{args.n_epochs}_depth_{args.branch_depth}_paranum_{args.num_branches}.pt')
