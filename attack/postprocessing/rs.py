import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

from attack.attack_interface import Postprocessing



class RandomizedSmoothingModel(nn.Module):
    def __init__(self, model, device, noise_std=0.25, num_samples=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.device = device

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for _ in range(self.num_samples):
                noise = torch.randn_like(x) * self.noise_std
                noise = noise.to(self.device)
                noisy_input = x + noise
                output = self.model(noisy_input)
                predicted_labels = output.argmax(dim=1)
                predictions.append(predicted_labels)
            predictions = torch.stack(predictions, dim=0)
            num_classes = output.size(1)
            class_counts = torch.stack([(predictions == i).sum(dim=0) for i in range(num_classes)], dim=1)
            class_frequencies = class_counts.float() / self.num_samples
        print(class_frequencies)
        return class_frequencies

class RandomizedSmoothing(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        noise_std = self.config.get("noise_std", 0.15)
        num_samples = self.config.get("num_samples", 100)
        device = self.args.device
        return RandomizedSmoothingModel(model, device, noise_std, num_samples)

