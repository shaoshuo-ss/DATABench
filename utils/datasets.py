# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, ImageNet, DatasetFolder
import numpy as np
import cv2
import os
from audit.utils import save_imagefolder, save_images_by_label


def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == 'mnist':
        # mean = torch.Tensor((0.1307))
        # var = torch.Tensor((0.3081))
        train_dataset = MNIST('./data/mnist/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(img_size),
                                  transforms.RandomHorizontalFlip(),
                                #   transforms.Normalize(mean.tolist(), var.tolist())
                              ]))
        test_dataset = MNIST('./data/mnist/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(img_size),
                                #  transforms.Normalize(mean.tolist(), var.tolist())
                             ]))
        num_classes = 10
        num_channels = 1
    elif dataset_name == 'cifar10':
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = CIFAR10('./data/cifar10/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.Normalize(mean.tolist(), var.tolist())
                                ]))
        test_dataset = CIFAR10('./data/cifar10/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                #    transforms.Normalize(mean.tolist(), var.tolist())
                               ]))
        num_classes = 10
        num_channels = 3
    elif dataset_name == 'cifar10-imagefolder':
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = ImageFolder('./data/cifar10-imagefolder/train/',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                ]))
        test_dataset = ImageFolder('./data/cifar10-imagefolder/test/',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                               ]))
        num_classes = 10
        num_channels = 3
    elif dataset_name == 'imagenet10':
        # mean = torch.Tensor((0.52283615, 0.47988218, 0.40605107))
        # var = torch.Tensor((0.29770654, 0.2888402, 0.31178293))
        full_dataset = ImageFolder('./data/imagenet10/train_set/',
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        # transforms.Normalize(mean.tolist(), var.tolist())
                                    ]))
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean.tolist(), var.tolist())
        ])
        num_classes = 10
        num_channels = 3
    elif dataset_name == "imagenet":
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = ImageNet("./data/imagenet/", split="train",
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Resize(img_size),
                                     transforms.Pad(32, padding_mode="reflect"),
                                     transforms.RandomCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                    #  transforms.Normalize(mean.tolist(), var.tolist())
                                 ]))
        test_dataset = ImageNet("./data/imagenet/", split="val",
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                #    transforms.Normalize(mean.tolist(), var.tolist())
                               ]))
        num_classes = 1000
        num_channels = 3
    elif dataset_name == "imagenet100":
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = ImageFolder("./data/benign_100/train/", 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        # transforms.Pad(32, padding_mode="reflect"),
                                        # transforms.RandomCrop(img_size),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.Normalize(mean.tolist(), var.tolist())
                                    ]))
        test_dataset = ImageFolder("./data/benign_100/val/", 
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Resize(img_size),
                                    #    transforms.Normalize(mean.tolist(), var.tolist())
                                   ]))
        num_classes = 100
        num_channels = 3
    elif dataset_name == "imagenet200":
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = ImageFolder("./data/sub-imagenet-200/train/", 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        transforms.Pad(32, padding_mode="reflect"),
                                        transforms.RandomCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.Normalize(mean.tolist(), var.tolist())
                                    ]))
        test_dataset = ImageFolder("./data/sub-imagenet-200/val/", 
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Resize(img_size),
                                    #    transforms.Normalize(mean.tolist(), var.tolist())
                                   ]))
        num_classes = 200
        num_channels = 3
    elif dataset_name == "vggface2":
        # mean = torch.Tensor((0.485, 0.456, 0.406))
        # var = torch.Tensor((0.229, 0.224, 0.225))
        train_dataset = ImageFolder("./data/vggface2/train/", 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        transforms.Pad(16, padding_mode="reflect"),
                                        transforms.RandomCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.Normalize(mean.tolist(), var.tolist())
                                    ]))
        test_dataset = ImageFolder("./data/vggface2/val/", 
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Resize(img_size),
                                    #    transforms.Normalize(mean.tolist(), var.tolist())
                                   ]))
        num_classes = 20
        num_channels = 3
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset, num_classes, num_channels


def get_dataset_root(dataset):
    if dataset == "cifar10-imagefolder":
        return "./data/cifar10-imagefolder/"
    elif dataset == "imagenet100":
        return "./data/benign_100/"
    else:
        raise NotImplementedError



class CustomImageFolder(ImageFolder):
    def __init__(self, original_dataset, indices):
        # Initialize a custom ImageFolder using a subset of indices
        self.samples = [original_dataset.samples[i] for i in indices]
        self.targets = [original_dataset.targets[i] for i in indices]
        self.class_to_idx = original_dataset.class_to_idx
        self.transform = original_dataset.transform
        self.target_transform = original_dataset.target_transform
        self.loader = original_dataset.loader
        self.classes = original_dataset.classes


class TensorImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        super().__init__()
        self.samples = images
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.samples[index], self.targets[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        return len(self.samples)

def split_imagefolder(dataset, ratios, dataset_name):
    """
    Split an ImageFolder dataset into three separate ImageFolder datasets.

    Parameters:
        dataset (ImageFolder): Original ImageFolder dataset
        ratios (tuple): Ratios for train, validation, and test split (e.g., (0.7, 0.15, 0.15))

    Returns:
        train_dataset, val_dataset, test_dataset (CustomImageFolder): Split datasets
    """
    assert sum(ratios) == 1.0, "The sum of ratios must be equal to 1.0"

    total_size = len(dataset)
    train_size = int(ratios[0] * total_size)
    val_size = int(ratios[1] * total_size)
    test_size = total_size - train_size - val_size  # Ensure total count matches

    dataset_root = get_dataset_root(dataset_name)

    train_path = os.path.join(dataset_root, "pub/")
    aux_path1 = os.path.join(dataset_root, "aux1/")
    aux_path2 = os.path.join(dataset_root, "aux2/")
    if not os.path.exists(train_path):
    
        train_indices, val_indices, test_indices = random_split(
            range(total_size), [train_size, val_size, test_size]
        )

        train_dataset = CustomImageFolder(dataset, train_indices)
        val_dataset = CustomImageFolder(dataset, val_indices)
        test_dataset = CustomImageFolder(dataset, test_indices)
        save_imagefolder(train_dataset, train_path)
        save_imagefolder(val_dataset, aux_path1)
        save_imagefolder(test_dataset, aux_path2)
    
    train_dataset = ImageFolder(root=train_path, 
                                transform=dataset.transform, 
                                target_transform=dataset.target_transform)
    val_dataset = ImageFolder(root=aux_path1, 
                                transform=dataset.transform, 
                                target_transform=dataset.target_transform)
    test_dataset = ImageFolder(root=aux_path2, 
                                transform=dataset.transform, 
                                target_transform=dataset.target_transform)

    # save the three datasets

    return train_dataset, val_dataset, test_dataset

# if __name__ == '__main__':
#     train_set, test_set = get_full_dataset('imagenet10', (224, 224))
#     # print(train_set.class_to_idx)
#     print(len(train_set))
#     print(len(test_set))
#     # print(train_set.targets)
