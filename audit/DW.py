#!~/Yanhengrui/anaconda3/envs/DVBW/bin python3
# -*- coding: utf-8 -*-

import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from torchvision.datasets import ImageFolder
import random
from typing import Dict, Tuple, List
from PIL import Image
import torchvision
import torch.nn.functional as F  
from audit.dataset_audit import DatasetAudit
from audit.dw_fun import fun
import argparse
import os



class DWImageFolder(ImageFolder):
    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class DW(DatasetAudit):
    """
    A class for dataset auditing, including watermark embedding, poisoning, and verification.
    """
    
    """
    Before running this code, please download the hard-to-generalize domain samples for CIFAR-10 from:
    https://www.dropbox.com/sh/hry5v7fxzzxcfr0/AADolCGag9DvY0RQaCzPsBVfa?dl=0
    and unzip the files to access the tensors required for the code.
    """

    def __init__(self, args):
        self.config = args.audit_config  
        self.gpus = args.gpus
        self.wm_data_path = self.config.get("poison_path")
        self.image_size = args.image_size
        self.dataset = args.dataset
        self.args = args
        os.makedirs(self.wm_data_path, exist_ok=True)
        self.device = torch.device(f"cuda:{self.gpus}" if torch.cuda.is_available() else "cpu")
        if args.dataset == 'cifar10-imagefolder':
            self.data_mean = torch.Tensor((0.485, 0.456, 0.406))
            self.data_std = torch.Tensor((0.229, 0.224, 0.225))
        elif args.dataset == 'imagenet100':
            self.data_mean = torch.Tensor((0.485, 0.456, 0.406))
            self.data_std = torch.Tensor((0.229, 0.224, 0.225))
        else:
            raise NotImplementedError


    def _create_args_from_config(self):
        args = argparse.Namespace()

        if self.config is not None:
            for key, value in self.config.items():
                setattr(args, key, value)
            
        setattr(args, 'gpus', self.gpus)
        return args

    

    def process_dataset(self, ori_dataset: ImageFolder, aux_dataset=None) -> Tuple[ImageFolder, Dict]:
        """
        Process the dataset by poisoning and embedding watermarks.

        Args:
            ori_dataset (ImageFolder): The original dataset.

        Returns:
            A tuple containing:
                - pub_dataset (ImageFolder): The processed dataset with poison and watermarks.
                - aux (dict): Auxiliary data required for verification.
        """
        args = self._create_args_from_config()
        if self.args.reprocessing:
            path = self.config.get("poison_path")
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            data_mean, data_std = torch.Tensor((0.485, 0.456, 0.406)), torch.Tensor((0.229, 0.224, 0.225))
            dw_ori_dataset = DWImageFolder(ori_dataset.root, transform=transforms.Compose([ori_dataset.transform, transforms.Normalize(data_mean, data_std)]), 
                                           target_transform=ori_dataset.target_transform)
            dw_ori_dataset.data_mean = data_mean
            dw_ori_dataset.data_std = data_std
            fun(dw_ori_dataset, args)

        pub_dataset = ImageFolder(
            os.path.join(self.wm_data_path, "train"), 
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(self.data_mean, self.data_std)
            ])
        )
        aux = {}
        
        return pub_dataset, aux

    
    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None, params: dict = None) -> float:
        """
        Conduct dataset auditing to a suspicious model and output the confidence value or p-value.

        Args:
            pub_dataset (ImageFolder): The processed dataset with embedded watermark.
            model: The model to be audited.
            aux (dict): Auxiliary data required for verification.
            params (dict): Additional parameters for verification.

        Returns:
            
        """

        tlabel = 4

        if self.dataset == 'cifar10-imagefolder':
            val_path = './tensor_val.pt'
        elif self.dataset == 'imagenet100':
            val_path = 'val_min_mi.pt'
        else:
            raise NotImplementedError()

        source_image = torch.load(val_path, weights_only=False)
        
        file_path = os.path.join(self.wm_data_path, "indices.pth")
        indices = torch.load(file_path, weights_only=False)
        source_image = torch.stack([source_image[index] for index in indices])
        source_image = source_image.to(self.device)

        value = torch.sum(model(source_image).max(1)[1] == tlabel) / len(source_image)

        print("Watermark Model's VSR:\n")
        print(value)
        return value

def get_dataset_auditing(args):
    if args.audit_method == "DW":
        return DW()
