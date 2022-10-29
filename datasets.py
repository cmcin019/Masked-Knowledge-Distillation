# Torch imports 
import torch

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision

def get_mnist_dataloaders():
    # Load training dataset
    train_dataset = datasets.MNIST(
        root = 'data/MNIST',
        train = True,
        transform = transforms.ToTensor(),
        download = True,
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=64, 
        shuffle=True
    )

    # Load validation dataset
    val_dataset = datasets.MNIST(
        root = 'data/MNIST',
        train = False,
        transform = transforms.ToTensor(),
        download = False,
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=64, 
        shuffle=True
    )

    return train_loader, val_loader

def get_cifar10_dataloaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)
            )
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=64,
        shuffle=True, 
        num_workers=2
    )

    # Load testing dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10',
        train=False,
        download=True, 
        transform=transform
    )
    testloader = DataLoader(
        testset, 
        batch_size=64,
        shuffle=False, 
        num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def get_imagenet2018_dataloaders():
    # Load training dataset
    train_dataset = datasets.ImageNet(
        root = 'data/image-net/ILSVRC',
        split = 'train',
        transform = transforms.ToTensor(),
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=64, 
        shuffle=True
    )

    # Load validation dataset
    val_dataset = datasets.ImageNet(
        root = 'data/image-net/ILSVRC',
        split = 'val',
        transform = transforms.ToTensor(),
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=64, 
        shuffle=True
    )

    return train_loader, val_loader

def get_imagenet2018_dataloaders2():
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)
            )
        ]
    )
    # Load training dataset
    train_dataset = datasets.ImageNet(
        root = '/media/cristopher/My Passport/image-net/ILSVRC',
        split = 'train',
        transform = transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=64, 
        shuffle=True
    )

    # Load validation dataset
    val_dataset = datasets.ImageNet(
        root = '/media/cristopher/My Passport/image-net/ILSVRC',
        split = 'val',
        transform = transform,
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=64, 
        shuffle=True
    )

    return train_loader, val_loader

def get_imagenet10k_dataloaders():
    pass

def get_imagenet1k_dataloaders():
    pass