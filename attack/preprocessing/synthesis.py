import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from audit.utils import save_images_by_label
import os
import logging
from tqdm import tqdm
from opacus import PrivacyEngine

from attack.attack_interface import Preprocessing
from utils.datasets import TensorImageDataset


def load_pretrained_model(num_classes=10):
    model = UNet2DModel.from_pretrained(
        "./model/ddpm-pretrained",
        class_embed_type="timestep",
        num_class_embeds=num_classes,
        resnet_time_scale_shift="default",
        low_cpu_mem_usage=False,
        device_map=None
    )
    return model


class ImageSynthesisPreprocessing(Preprocessing):
    def process(self, dataset, aux_dataset=None):
        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        ddpm_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])
        dataset.transform = ddpm_transforms
        bs = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 10)
        lr = self.config.get("lr", 1e-4)
        timesteps = self.config.get("timesteps", 1000)

        logger = logging.getLogger(__name__)
        ddpm_path = self.config.get("ddpm_path", "./model/ddpm_{}_{}.pth".format(self.args.dataset, self.args.audit_method))

        if not os.path.exists(ddpm_path) or self.config.get("retrain", True):
            # retrain the diffusion model
            model = load_pretrained_model(self.args.num_classes)
            logger.info("Load pre-trained DDPM and fine-tune.")
            os.makedirs(os.path.dirname(ddpm_path), exist_ok=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, 
                                                           num_training_steps=(len(dataloader) * epochs))
            model = model.to(self.device)
            scheduler = DDPMScheduler(num_train_timesteps=timesteps)

            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0.0
                for x_0, labels in dataloader:
                    x_0, labels = x_0.to(self.device), labels.to(self.device)
                    noise = torch.randn_like(x_0)
                    
                    # Sample timesteps
                    timestep = torch.randint(0, timesteps, (x_0.size(0),), device=self.device)
                    noisy_x = scheduler.add_noise(x_0, noise, timestep)

                    optimizer.zero_grad()
                    loss = F.mse_loss(model(noisy_x, timestep, class_labels=labels, return_dict=False)[0], noise)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    total_loss += loss.item()

                logger.info(f"Epoch [{epoch+1}/{epochs}], DDPM Loss: {total_loss/len(dataloader):.4f}")

            # Save the model
            torch.save(model.state_dict(), ddpm_path)
        else:
            # load the pre-trained diffusion model
            model = load_pretrained_model(self.args.num_classes)
            logger.info("Load pre-trained DDPM")
            model.load_state_dict(torch.load(ddpm_path))
            model = model.to(self.device)
        
        # Generate synthetic datasets using DDPM
        num_samples_per_class = self.config.get("num_samples_per_class", 5000)
        model.eval()
        images = []
        targets = []
        scheduler = DDPMScheduler(num_train_timesteps=timesteps)

        denormalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in std]),
            transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.])
        ])
        targets = [torch.tensor([i] * num_samples_per_class) for i in range(self.args.num_classes)]
        targets = torch.concat(targets).to(self.device)
        # print(targets.shape)
        targets_dataloader = DataLoader(TensorDataset(targets), batch_size=bs, shuffle=False)
        with torch.no_grad():
            for target in tqdm(targets_dataloader):
                x_t = torch.randn((target[0].size(0), 3, 
                                   self.args.image_size, self.args.image_size)).to(self.device)
                for t in scheduler.timesteps:
                    model_output = model(x_t, t, class_labels=target[0]).sample
                    x_t = scheduler.step(model_output, t, x_t).prev_sample
                # images.append(denormalize_transform(x_t))
                images.append(torch.clamp(x_t, 0, 1))
                # images.append(x_t)
                # targets.append(torch.ones(x_t.size(0)) * i)
        images = torch.concat(images)
        # targets = targets

        transform = transforms.Compose([
            # dataset.transform,
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])
        processed_dataset = TensorImageDataset(images.cpu(), targets.cpu())
        save_images_by_label(processed_dataset, f"./data/synthesis/{self.args.audit_method}/{self.args.dataset}/", dataset.classes)
        processed_dataset.transform = transform

        return processed_dataset


class DPImageSynthesisPreprocessing(Preprocessing):
    def process(self, dataset, aux_dataset=None):
        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        ddpm_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])
        dataset.transform = ddpm_transforms
        bs = self.config.get("batch_size", 64)
        test_bs = self.config.get("test_batch_size", 64)
        epochs = self.config.get("epochs", 10)
        lr = self.config.get("lr", 1e-4)
        timesteps = self.config.get("timesteps", 1000)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        target_epsilon = self.config.get("target_epsilon", 8.0)
        target_delta = self.config.get("target_delta", 1e-5)

        logger = logging.getLogger(__name__)
        ddpm_path = self.config.get("ddpm_path", "./model/ddpm_{}_{}.pth".format(self.args.dataset, self.args.audit_method))

        if not os.path.exists(ddpm_path) or self.config.get("retrain", True):
            # retrain the diffusion model
            model = load_pretrained_model(self.args.num_classes)
            logger.info("Load pre-trained DDPM and fine-tune.")
            os.makedirs(os.path.dirname(ddpm_path), exist_ok=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, 
                                                           num_training_steps=(len(dataloader) * epochs))
            model = model.to(self.device)
            model.train()
            scheduler = DDPMScheduler(num_train_timesteps=timesteps)
            
            privacy_engine = PrivacyEngine()
            model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
                epochs=epochs,
                poisson_sampling=False
            )

            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0.0
                for x_0, labels in dataloader:
                    x_0, labels = x_0.to(self.device), labels.to(self.device)
                    noise = torch.randn_like(x_0)
                    
                    # Sample timesteps
                    timestep = torch.randint(0, timesteps, (x_0.size(0),), device=self.device)
                    noisy_x = scheduler.add_noise(x_0, noise, timestep)

                    optimizer.zero_grad()
                    loss = F.mse_loss(model(noisy_x, timestep, class_labels=labels, return_dict=False)[0], noise)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    total_loss += loss.item()

                logger.info(f"Epoch [{epoch+1}/{epochs}], DDPM Loss: {total_loss/len(dataloader):.4f}")

            # Save the model
            torch.save(model.state_dict(), ddpm_path)
        else:
            # load the pre-trained diffusion model
            model = load_pretrained_model(self.args.num_classes)
            logger.info("Load pre-trained DDPM")
            model.load_state_dict(torch.load(ddpm_path))
            model = model.to(self.device)
        
        # Generate synthetic datasets using DDPM
        num_samples_per_class = self.config.get("num_samples_per_class", 5000)
        model.eval()
        images = []
        targets = []
        scheduler = DDPMScheduler(num_train_timesteps=timesteps)

        denormalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in std]),
            transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.])
        ])
        targets = [torch.tensor([i] * num_samples_per_class) for i in range(self.args.num_classes)]
        targets = torch.concat(targets).to(self.device)
        # print(targets.shape)
        targets_dataloader = DataLoader(TensorDataset(targets), batch_size=test_bs, shuffle=False)
        with torch.no_grad():
            for target in tqdm(targets_dataloader):
                x_t = torch.randn((target[0].size(0), 3, 
                                   self.args.image_size, self.args.image_size)).to(self.device)
                for t in scheduler.timesteps:
                    model_output = model(x_t, t, class_labels=target[0]).sample
                    x_t = scheduler.step(model_output, t, x_t).prev_sample
                # images.append(denormalize_transform(x_t))
                images.append(torch.clamp(x_t, 0, 1))
                # images.append(x_t)
                # targets.append(torch.ones(x_t.size(0)) * i)
        images = torch.concat(images)
        # targets = targets

        transform = transforms.Compose([
            # dataset.transform,
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])
        processed_dataset = TensorImageDataset(images.cpu(), targets.cpu())
        save_images_by_label(processed_dataset, f"./data/synthesis/{self.args.audit_method}/{self.args.dataset}/", dataset.classes)
        processed_dataset.transform = transform

        return processed_dataset
