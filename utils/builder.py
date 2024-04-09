import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from utils.util import CLDataTransform
from torch.utils.data import DataLoader
from randaugment import CIFAR10Policy, ImageNetPolicy, Cutout, RandAugment
from utils.noisy_cifar import ClenCIFAR100, NoisyCIFAR100
from utils.image_folder import IndexedImageFolder
# from data.food101 import Food101
# from data.food101n import Food101N
import numpy as np
import cv2

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class GaussianBlur(object):
    def __init__(self, kernel_size, minimum=0.1, maximum=2.0):
        self.min = minimum
        self.max = maximum
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


# dataset --------------------------------------------------------------------------------------------------------------------------------------------
def build_transform(rescale_size=512, crop_size=448):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=crop_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=crop_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform, 'train_strong_aug': train_transform_strong_aug,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform, 'cifar_train_strong_aug': cifar_train_transform_strong_aug}


# def build_cifar10n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
#     train_data = NoisyCIFAR10(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
#                                openset_ratio=openset_ratio, verbose=False)
#     test_data = NoisyCIFAR10(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
#                               openset_ratio=openset_ratio, verbose=False)
#     return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


def CLDTransform_cifar(sample):
    transformation = build_transform(32, 32)
    transform_weak, transform_strong = transformation['cifar_train'], transformation['cifar_train_strong_aug']
    x_w1 = transform_weak(sample)
    x_w2 = transform_weak(sample)
    x_s = transform_strong(sample)
    return x_w1, x_w2, x_s



def build_cifar100c_dataset(root, train_transform, test_transform, openset_ratio):
    train_data = ClenCIFAR100(root, train=True, transform=train_transform, download=False, openset_ratio=openset_ratio)
    test_data = ClenCIFAR100(root, train=False, transform=test_transform, download=False, openset_ratio=openset_ratio)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}

def build_cifar100n_dataset(root, train_transform, test_transform, openset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=False, openset_ratio=openset_ratio)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=False, openset_ratio=openset_ratio)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


# def build_webfg_dataset(root, train_transform, test_transform):
#     train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform)
#     test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
#     return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}
#
#
# def build_food101n_dataset(root, train_transform, test_transform):
#     train_data = Food101N(root, transform=train_transform)
#     test_data = Food101(os.path.join(root, 'food-101'), split='test', transform=test_transform)
#     return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}



def build_cifarn_dataset_loader(cfg):
    assert cfg.dataset.startswith('cifar')
    transformation = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    if cfg.dataset == 'cifar100':
        dataset_1 = build_cifar100c_dataset(root=os.path.join(cfg.database, cfg.dataset),
                                          train_transform=CLDTransform_cifar,
                                          test_transform=transformation['cifar_test'],
                                          openset_ratio=cfg.openset_ratio)
        dataset_2 = build_cifar100n_dataset(root=os.path.join(cfg.database, cfg.dataset),
                                          train_transform=CLDTransform_cifar,
                                          test_transform=transformation['cifar_test'],
                                          openset_ratio=cfg.openset_ratio)
    else:
        raise AssertionError(f'{cfg.dataset} dataset is not supported yet.')
    # dataset 1
    train_dataset_1, test_dataset = dataset_1['train'], dataset_1['test']
    train_sampler_1 = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset_1)
    train_loader_1 = DataLoader(dataset=train_dataset_1,
                              batch_size=cfg.batch_size,
                              shuffle=(train_sampler_1 is None),
                              sampler=train_sampler_1,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)
    
    # dataset 2
    train_dataset_2 = dataset_2['train']
    train_sampler_2 = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset_2)
    train_loader_2 = DataLoader(dataset=train_dataset_2,
                              batch_size=cfg.batch_size,
                              shuffle=(train_sampler_2 is None),
                              sampler=train_sampler_2,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg.batch_size * 4,
                             shuffle=False,
                             sampler=torch.utils.data.distributed.DistributedSampler(dataset=test_dataset, shuffle=False),
                             num_workers=8,
                             pin_memory=False)
    return dataset_1, train_loader_1, train_sampler_2, dataset_2, train_loader_2, train_sampler_2, test_loader



# optimizer, scheduler -------------------------------------------------------------------------------------------------------------------------------
def build_sgd_optimizer(params, lr, weight_decay, nesterov=True):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

