import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def load_cifar10(batch_size,
                 data_dir='./data',
                 val_size=0.1,
                 augment=True,
                 shuffle=True,
                 seed=1):
    """Load CIFAR-10 data into train/val/test data loader"""

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_workers = 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                5, translate=(0.1, 0.1), scale=(0, 0.1), shear=5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transform

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
