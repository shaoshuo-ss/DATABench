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


def replace_last_layer_with_identity(model):
    """
    Replace the last layer of the model with nn.Identity and return the modified model.
    
    Args:
        model (nn.Module): The input model.
        
    Returns:
        model (nn.Module): The modified model (last layer replaced with Identity).
    """
    # Get all submodules of the model
    modules = list(model.named_modules())
    
    # Find the last layer
    if len(modules) < 2:
        raise ValueError("The model must contain at least two layers (input and output layers).")
    
    # Name and module of the last layer
    last_layer_name, last_layer = modules[-1]
    
    print(f"Found the last layer: {last_layer_name} -> {last_layer}")
    
    # Replace the last layer with Identity
    *parent_names, layer_name = last_layer_name.split('.')
    parent_module = model
    for name in parent_names:
        parent_module = getattr(parent_module, name)
    setattr(parent_module, layer_name, nn.Identity())
    
    print("The last layer has been replaced with Identity.")
    
    return model


class SVMOutlierDetectionModel(nn.Module):
    def __init__(self, model, dataset, device, config, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        self.args = arguments
        self.feature_extractor = copy.deepcopy(self.model)
        self.feature_extractor = replace_last_layer_with_identity(self.feature_extractor)
        nu = self.config.get("nu", 0.1)
        self.detector = self._train_detector(nu)

    def _extract_features(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.cpu().detach().numpy()
    
    def _train_detector(self, nu=0.1):
        # Get features
        dataloader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True)
        features = []
        for images, _ in dataloader:
            images = images.to(self.device)
            output_features = self._extract_features(images)
            features.append(output_features)
        features = np.vstack(features)

        # Train the one-class-SVM as outlier detector
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma='auto')
        clf.fit(features)
        return clf
    
    def forward(self, x):
        features = self._extract_features(x)
        results = self.detector.predict(features)
        preds = self.model(x)
        # print(results)
        for idx in range(len(results)):
            if results[idx] == -1:
                preds[idx] = torch.zeros_like(preds[idx])
        return preds


class KNNOutlierDetectionModel(nn.Module):
    def __init__(self, model, dataset, device, config, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        self.args = arguments
        self.feature_extractor = copy.deepcopy(self.model)
        self.feature_extractor = replace_last_layer_with_identity(self.feature_extractor)
        self.k = self.config.get("k", 5)
        self.confidence = self.config.get("confidence", 0.99)
        self.detector, self.threshold = self._train_detector(self.k, self.confidence)
        print(self.threshold)

    def _extract_features(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.cpu().detach().numpy()
    
    def _train_detector(self, k, confidence):
        # Get features
        dataloader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True)
        features = []
        for images, _ in dataloader:
            images = images.to(self.device)
            output_features = self._extract_features(images)
            features.append(output_features)
        features = np.vstack(features)

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(features)
        distances, _ = knn.kneighbors(features)
        avg_distances = distances.mean(axis=1)

        mu, sigma = np.mean(avg_distances), np.std(avg_distances)
        threshold = norm.ppf(confidence, loc=mu, scale=sigma)
        return knn, threshold
    
    def forward(self, x):
        features = self._extract_features(x)
        results = self.detector.kneighbors(features)[0].mean(axis=1)
        results = results > self.threshold
        preds = self.model(x)
        for idx in range(len(results)):
            if results[idx]:
                preds[idx] = torch.zeros_like(preds[idx])
        return preds



class SVMOutlierDetection(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        return SVMOutlierDetectionModel(model, aux_dataset, self.args.device, self.config, self.args)
    

class KNNOutlierDetection(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        return KNNOutlierDetectionModel(model, aux_dataset, self.args.device, self.config, self.args)
