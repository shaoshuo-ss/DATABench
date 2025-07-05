import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm, lognorm
import copy

from attack.attack_interface import Postprocessing

import torch
import torch.nn as nn


def get_last_layer_and_replace_with_identity(model):
    """
    Extract the last meaningful layer of the model (e.g., classifier) and return two models:
    - The extracted last layer.
    - The modified model with the last layer replaced by nn.Identity.
    
    Args:
        model (nn.Module): The input model.
    
    Returns:
        nn.Module: The extracted last layer.
        nn.Module: The modified model with Identity replacing the last layer.
    """
    temp_model = copy.deepcopy(model)
    modules = list(temp_model.named_modules())
    
    if len(modules) < 2:
        raise ValueError("The model must contain at least two layers (input and output layers).")
    
    # Traverse the modules in reverse to find the first non-Identity layer
    for i in range(len(modules) - 1, -1, -1):
        layer_name, layer = modules[i]
        if not isinstance(layer, nn.Identity):
            print(f"Found the last meaningful layer: {layer_name} -> {layer}")
            
            # Get parent module
            *parent_names, layer_name = layer_name.split('.')
            parent_module = temp_model
            for name in parent_names:
                parent_module = getattr(parent_module, name)
            
            # Replace the identified layer with Identity
            setattr(parent_module, layer_name, nn.Identity())
            
            print("The last meaningful layer has been replaced with Identity.")
            return temp_model, layer
    
    raise ValueError("No meaningful layer found to replace. The model might consist only of Identity layers.")


class OutputNoiseModel(nn.Module):
    def __init__(self, model, device, config, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.config = config
        self.args = arguments
        self.sigma = self.config.get("sigma", 0.1)
    
    def forward(self, x):
        preds = self.model(x)
        noise = torch.normal(0, self.sigma, size=preds.size()).to(self.device)
        preds = preds + noise
        return preds
    
class OutputNoise(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        return OutputNoiseModel(model, self.args.device, self.config, self.args)
    

class FeatureNoiseModel(nn.Module):
    def __init__(self, model, device, config, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.config = config
        self.args = arguments
        self.sigma = self.config.get("sigma", 0.1)
        self.feature_extractor, self.classifier = get_last_layer_and_replace_with_identity(self.model)
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

    def forward(self, x):
        features = self.feature_extractor(x)
        noise = torch.normal(0, self.sigma, size=features.size()).to(self.device)
        features = features + noise
        preds = self.classifier(features)
        return preds
    

class FeatureNoise(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        return FeatureNoiseModel(model, self.args.device, self.config, self.args)
