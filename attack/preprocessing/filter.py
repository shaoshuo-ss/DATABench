from attack.attack_interface import Preprocessing
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import pywt


class MedianFilterPreprocessing(Preprocessing):
    def process(self, dataset, aux_dataset=None):
        def median_filter(img):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img_np = np.array(img)
            filtered_img = cv2.medianBlur(img_np, self.config['kernel_size'])
            return transforms.ToTensor()(filtered_img)
        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        transform = transforms.Compose([
            dataset.transform,
            transforms.Lambda(median_filter),
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
        ])

        dataset.transform = transform
        return dataset

class GaussianFilterPreprocessing(Preprocessing):
    def process(self, dataset, aux_dataset=None):
        def gaussian_filter(img):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img_np = np.array(img)
            filtered_img = cv2.GaussianBlur(img_np, (self.config['kernel_size'], self.config['kernel_size']), self.config['sigma'])
            return transforms.ToTensor()(filtered_img)

        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        transform = transforms.Compose([
            dataset.transform,
            transforms.Lambda(gaussian_filter),
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])

        dataset.transform = transform
        return dataset


class WaveletFilterPreprocessing(Preprocessing):
    def process(self, dataset, aux_dataset=None):
        def wavelet_filter(img):
            if isinstance(img, torch.Tensor):
                np_image = img.numpy()
                coeffs = pywt.wavedec2(np_image, self.config.get("wavelet", "haar"), level=self.config['level'], axes=(1, 2))
                coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs, axes=(1, 2))

                # calculate threshold
                threshold = self.config.get("threshold", 0.1) * np.max(coeffs_arr)
                coeffs_arr = pywt.threshold(coeffs_arr, threshold, mode='soft')
                coeffs = pywt.array_to_coeffs(coeffs_arr, coeffs_slices, output_format='wavedec2')
                img = pywt.waverec2(coeffs, self.config.get("wavelet", "haar"), axes=(1, 2))
                img = np.clip(img, 0, 1)
                img = torch.tensor(img, dtype=torch.float32)
                return img

        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        transform = transforms.Compose([
            dataset.transform,
            transforms.Lambda(wavelet_filter),
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])

        dataset.transform = transform
        return dataset
