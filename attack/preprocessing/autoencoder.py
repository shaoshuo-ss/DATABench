from attack.attack_interface import Preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasets import TensorImageDataset
import numpy as np
import os
import logging
from tqdm import tqdm


class AutoEncoder(nn.Module):
    def __init__(self, image_size):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderPreprocessing(Preprocessing):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = self.config.get("autoencoder_path")
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.autoencoder = AutoEncoder(args.image_size).to(self.device)
        self._load_autoencoder()
    
    def _train_autoencoder(self, aux_dataset):
        epochs = self.config.get("epochs", 150)
        batch_size = self.config.get("batch_size", 16)
        lr = self.config.get("lr", 0.01)
        dataloader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # criterion = nn.MSELoss()
        criterion = nn.BCELoss()
        logger = logging.getLogger(__name__)
        
        self.autoencoder.train()
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            for images, _ in dataloader:
                images = images.to(self.device)
                optimizer.zero_grad()
                outputs = self.autoencoder(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            schedule.step()
        logger.info("Autoencoder training completed. Saving model...")
        torch.save(self.autoencoder.state_dict(), self.model_path)

    def _load_autoencoder(self):
        logger = logging.getLogger(__name__)
        if os.path.exists(self.model_path) and not self.config.get("retrain"):
            logger.info("Loading pre-trained autoencoder...")
            self.autoencoder.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.autoencoder.eval()
        else:
            logger.info("No pre-trained autoencoder found. Training required.")

    def process(self, dataset, aux_dataset=None):
        logger = logging.getLogger(__name__)
        if (aux_dataset is not None and not os.path.exists(self.model_path)) or self.config.get("retrain"):
            self._train_autoencoder(aux_dataset)
        
        dataloader = DataLoader(dataset, batch_size=self.args.bs, shuffle=False)
        processed_images = []
        labels = []
        self.autoencoder.eval()
        with torch.no_grad():
            for images, label in dataloader:
                images = images.to(self.device)
                # processed_image = self.autoencoder(images)
                processed_images.append(self.autoencoder(images).detach().cpu())
                labels.append(label)
        
        # processed_dataset = [(img, lbl) for img, lbl in zip(processed_images, labels)]
        processed_images = torch.cat(processed_images, 0)
        labels = torch.cat(labels, 0)
        # print(processed_images.shape)
        # print(labels.shape)
        # exit(0)
        # processed_dataset = torch.utils.data.TensorDataset(processed_images, labels)
        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        processed_dataset = TensorImageDataset(processed_images, labels, 
                                               transform=transforms.Compose([
                                                    transforms.Resize(self.args.image_size),
                                                    transforms.RandomResizedCrop(self.args.image_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.Normalize(mean, std)
                                               ]), 
                                               target_transform=dataset.target_transform)
        return processed_dataset
