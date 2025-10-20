import logging

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from randomaug import RandAugment
import os
os.environ['TORCHVISION_DATASETS_MIRRORS'] = '["http://mirror.example.com/torchvision/datasets"]'
logger = logging.getLogger(__name__)


def get_loader(args):
    
    transform_train = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.CenterCrop((args.size, args.size)),
    transforms.RandomCrop(args.size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ])
    
    transform_train_gray = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.CenterCrop((args.size, args.size)),
    transforms.RandomCrop(args.size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.2)),
    ])

    transform_test_gray = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.2)),
        ])


    if args.aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))
        transform_train_gray.transforms.insert(0, RandAugment(N, M))

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.data_dir, 
                                    train=True, 
                                    download=True, 
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, 
                                   train=False, 
                                   download=True, 
                                   transform=transform_test)

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir,
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
    elif args.dataset == "FashionMNIST":
        trainset = datasets.FashionMNIST(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train_gray)
        testset = datasets.FashionMNIST(root=args.data_dir,
                                    train=False,
                                    download=False,        
                                    transform=transform_test_gray) 
    

    num_val_percent = 0.1
    num_val = int(len(trainset)*num_val_percent)
    num_train = len(trainset) - num_val
    trainset, valset = torch.utils.data.random_split(trainset, [num_train, num_val])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=8,persistent_workers=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=True, num_workers=8,persistent_workers=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=8,persistent_workers=False)
    return trainloader, valloader, testloader
