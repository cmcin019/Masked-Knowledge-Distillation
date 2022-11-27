# Torch imports 
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision

from PIL import Image
import utils
from masking_generator import MaskingGenerator

class DataAugmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

        self.masked_position_generator = MaskingGenerator(
            (4, 4), num_masking_patches=75,
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops, self.masked_position_generator()

def get_mnist_dataloaders(args):
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )

    # Load training dataset
    train_dataset = datasets.MNIST(
        root = 'data/MNIST',
        train = True,
        transform = transform,
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

def get_cifar10_dataloaders(args):
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #         transforms.Normalize(
    #             (0.5, 0.5, 0.5), 
    #             (0.5, 0.5, 0.5)
    #         )
    #     ]
    # )

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
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
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

def get_imagenet2018_dataloaders(args):
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

def get_imagenet2018_dataloaders2(args):
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

def get_imagenet10k_dataloaders(args):
    pass

def get_imagenet1k_dataloaders(args):
    pass