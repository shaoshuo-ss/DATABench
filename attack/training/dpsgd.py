import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from attack.attack_interface import Training
from utils.test import test_img
from tqdm import tqdm
import logging
import os
import numpy as np


class LayerNorm2d(nn.Module):
    """
    LayerNorm for [N, C, H, W] tensors. Applies LayerNorm over the channel dimension.
    """
    def __init__(self, num_channels, eps=1e-5, affine=False):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # x: [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # back: [N, H, W, C] -> [N, C, H, W]
        return x.permute(0, 3, 1, 2)


def replace_bn_with_gn(model, group=32):
    """
    Recursively replace all nn.BatchNorm2d layers in the model with nn.GroupNorm.
    
    Args:
        model (nn.Module): The input PyTorch model.
        num_groups (int): Number of groups to use for GroupNorm. Default is 32.
        
    Returns:
        nn.Module: The model with BatchNorm2d replaced by GroupNorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Get the number of channels from the BatchNorm layer
            num_channels = module.num_features
            group_num = min(group, num_channels)
            while num_channels % group_num != 0:
                group_num -= 1
            
            ln = nn.GroupNorm(num_groups=group_num, num_channels=num_channels)
            setattr(model, name, ln)
        else:
            replace_bn_with_gn(module)
    # return model


def replace_bn_with_identity(model):
    """
    Recursively replace all nn.BatchNorm2d layers in the model with nn.GroupNorm.
    
    Args:
        model (nn.Module): The input PyTorch model.
        num_groups (int): Number of groups to use for GroupNorm. Default is 32.
        
    Returns:
        nn.Module: The model with BatchNorm2d replaced by GroupNorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            
            ln = nn.Identity()
            setattr(model, name, ln)
        else:
            replace_bn_with_identity(module)


def check_grad_sample_dims(model):
    inconsistent_layers = []
    batch_sizes = set()
    for name, param in model.named_parameters():
        if param.grad_sample is not None:
            bs = param.grad_sample.shape[0]
            batch_sizes.add(bs)
            print(f"Layer: {name}, Gradient Sample Batch Size: {bs}")
    if len(batch_sizes) > 1:
        print(f"Error: Inconsistent batch sizes {batch_sizes}")
    else:
        print(f"All gradients have batch size {batch_sizes.pop()}")


class DPSGD(Training):
    def train(self, train_dataset, test_dataset, model, aux_dataset=None):
        # Clear the data augmentation in dataset (Note: DP-SGD can be regarded as a strong data augmentation method)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.args.image_size)
        ])
        train_dataset.transform = transform

        # Extract parameters
        batch_size = self.config.get("batch_size", 128)
        epochs = self.config.get("epochs", 90)
        lr = self.config.get("lr", 0.01)
        wd = self.config.get("wd", 0.0)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        target_epsilon = self.config.get("target_epsilon", 8.0)
        target_delta = self.config.get("target_delta", 1e-5)
        momentum = self.config.get("momentum", 0.9)
        # grad_accumulation = self.config.get("grad_accumulation", 1)
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # validate model
        # errors = ModuleValidator.validate(model, strict=False)
        # if len(errors) > 0:
            # model = ModuleValidator.fix(model)
        # replace_bn_with_gn(model)
        # replace_bn_with_identity(model)
        # for name, module in model.named_children():
        #     module = ModuleValidator.fix(module)
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        logger = logging.getLogger(__name__)
        # Enable PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            poisson_sampling=False
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # print(f"Using DP-SGD with ε = {privacy_engine.get_epsilon(target_delta)}")

        # Training loop
        model.train()
        model = model.to(self.device)
        acc_best = None
        es_count = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            epoch_loss = []
            accumulation_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss = loss
                loss.backward()
                # gradient accumulation
                # accumulation_count += 1
                # if accumulation_count == grad_accumulation:
                # check_grad_sample_dims(model)
                optimizer.step()
                    # accumulation_count = 0
                epoch_loss.append(loss.item())

                # if batch_idx % 100 == 0:
                    # logger.info(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            logger.info(f"Epoch {epoch + 1}/{epochs}: loss: {np.mean(epoch_loss):.4f}")
            scheduler.step()
            # Model evaluation
            if (epoch + 1) % self.args.eval_rounds == 0:
                acc_val, loss_val = test_img(model, test_dataset, self.args)
                logger.info("Epoch {} val loss:{:.4f}, val acc:{:.3f}".format(epoch + 1, loss_val, acc_val))
                if acc_best is None or acc_best < acc_val:
                    acc_best = acc_val
                    if self.args.save_model:
                        torch.save(model.module.state_dict(), os.path.join(self.args.save_path, "model_best.pth"))
                    es_count = 0
                else:
                    es_count += 1
                    if es_count >= self.args.stopping_rounds:
                        break
        if self.args.save_model:
            torch.save(model.module.state_dict(), os.path.join(self.args.save_path, "model_last_epochs_" + str(epoch) + ".pth"))
            model.module.load_state_dict(torch.load(os.path.join(self.args.save_path, "model_best.pth")))
            acc_test, _ = test_img(model, test_dataset, self.args)
            logger.info("Best Testing Accuracy:{:.2f}".format(acc_test))
        return model
