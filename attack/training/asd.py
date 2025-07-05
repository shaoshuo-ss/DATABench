import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import numpy as np
import copy
from tqdm import tqdm
from copy import deepcopy
import random
import logging

from attack.attack_interface import Training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import time
from PIL import Image

from utils.test import test_img
from utils.train import get_optim
import os
# import timm


class CustomConcatDataset(Dataset):
    def __init__(self, datasets):
        self.data = []
        self.targets = []
        self.transform = datasets[0].transform  # Use the transform of the first dataset
        self.target_transform = datasets[0].target_transform
        self.classes = datasets[0].classes  # Use the class information
        self.class_to_idx = datasets[0].class_to_idx

        for dataset in datasets:
            for path, label in dataset.samples:
                self.data.append(path)  # Only store paths to save memory
                self.targets.append(label)
        
        # self.data = torch.tensor(self.data)  # Optional
        # self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        img_path = self.data[index]  # Read the path
        image = Image.open(img_path).convert("RGB")  # Read the image
        if self.transform:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.data)


class PoisonLabelDataset(Dataset):
    """Poison-Label dataset wrapper.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """

    def __init__(self, dataset, transform, poison_idx):
        super(PoisonLabelDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        self.train = self.dataset.train
        if self.train:
            self.data = self.dataset.data
            self.targets = self.dataset.targets
            self.poison_idx = poison_idx
        # else:
        #     # Only fetch poison data when testing.
        #     self.data = self.dataset.data[np.nonzero(poison_idx)[0]]
        #     self.targets = self.dataset.targets[np.nonzero(poison_idx)[0]]
        #     self.poison_idx = poison_idx[poison_idx == 1]
        self.transform = transform
        # self.pre_transform = self.dataset.pre_transform
        # self.primary_transform = self.dataset.primary_transform
        # self.remaining_transform = self.dataset.remaining_transform
        # self.prefetch = self.dataset.prefetch
        # if self.prefetch:
        #     self.mean, self.std = self.dataset.mean, self.dataset.std

        # self.bd_transform = transform
        # self.target_label = target_label

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            with open(self.data[index], "rb") as f:
                img = np.array(Image.open(f).convert("RGB"))
        else:
            img = self.data[index]
        target = self.targets[index]
        poison = 0
        origin = target  # original target

        if self.poison_idx[index] == 1:
            # img = self.bd_first_augment(img, bd_transform=self.bd_transform)
            # target = self.target_label
            poison = 1
        # else:
            # img = self.bd_first_augment(img, bd_transform=None)
        img = self.transform(img)
        item = {"img": img, "target": target, "poison": poison, "origin": origin}

        return item

    def __len__(self):
        return len(self.data)


class LinearModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(LinearModel, self).__init__()
        self.backbone = backbone
        self.linear = classifier

    def forward(self, x):
        feature = self.backbone(x)
        out = self.linear(feature)
        return out

    def update_encoder(self, backbone):
        self.backbone = backbone



class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""

    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss

        return loss


class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u

class Record(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = torch.zeros(self.size)

    def update(self, batch_data):
        self.data[self.ptr : self.ptr + len(batch_data)] = batch_data
        self.ptr += len(batch_data)

def poison_linear_record(model, loader, criterion):
    num_data = len(loader.dataset)
    target_record = Record("target", num_data)
    poison_record = Record("poison", num_data)
    origin_record = Record("origin", num_data)
    loss_record = Record("loss", num_data)
    feature_record = Record("feature", (num_data, model.linear.in_features))
    record_list = [
        target_record,
        poison_record,
        origin_record,
        loss_record,
        feature_record,
    ]

    model.eval()
    gpu = next(model.parameters()).device
    for _, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            feature = model.backbone(data)
            output = model.linear(feature)
        criterion.reduction = "none"
        raw_loss = criterion(output, target)

        target_record.update(batch["target"])
        poison_record.update(batch["poison"])
        origin_record.update(batch["origin"])
        loss_record.update(raw_loss.cpu())
        feature_record.update(feature.cpu())

    return record_list

def class_aware_loss_guided_split(record_list, has_indice, all_data_info, choice_num, logger):
    """
    Implements class-aware loss-guided splitting (Phase 1 of ASD).
    Selects a fixed number (`choice_num`) of samples with the lowest loss 
    *per class* from the remaining candidates (`all_data_info`) and adds 
    them to the current clean pool (`has_indice`).

    Args:
        record_list (list): List of metrics recorded by poison_linear_record 
                            (must include 'loss' and 'poison').
        has_indice (np.array): Indices of samples already in the clean pool (Dc).
        all_data_info (dict): Dictionary mapping class labels (str) to lists of 
                              candidate sample indices not yet in Dc.
        choice_num (int): Number of samples to select per class in this step.
        logger: Logger instance.

    Returns:
        np.array: A boolean mask indicating the updated clean pool (Dc).
                  1 if a sample is in Dc, otherwise 0.
    """
    keys = [r.name for r in record_list] # Retrieve metric names ('loss', 'poison', etc.)
    loss = record_list[keys.index("loss")].data.numpy() # Extract loss values of all samples
    poison = record_list[keys.index("poison")].data.numpy() # Extract poisoning flags (0 or 1)
    # Initialize the mask for the clean pool (initially all zeros)
    clean_pool_idx = np.zeros(len(loss))

    # Start with indices already selected (from seed samples or previous iterations)
    total_indice = has_indice.tolist()
    # Iterate over candidate samples for each class
    for k, v in all_data_info.items():
        v = np.array(v) # Convert the list of indices for this class into a numpy array
        if len(v) == 0: continue # Skip if there are no remaining candidates for this class
        loss_class = loss[v] # Get the loss values of these candidate samples
        # Find indices (relative to `v`) of the `choice_num` lowest-loss samples
        # Ensure that we do not attempt to select more samples than available
        num_to_select = min(choice_num, len(v))
        indice_class = loss_class.argsort()[: num_to_select]
        # Retrieve actual dataset indices corresponding to the selected relative indices (from `v`)
        indice = v[indice_class]
        # Add these newly selected indices to the total list of clean pool indices
        total_indice += indice.tolist()

    # Convert the final list of indices into a numpy array
    total_indice = np.array(total_indice, dtype=int) # Ensure integer type
    # Update the boolean mask: set elements corresponding to selected indices to 1
    clean_pool_idx[total_indice] = 1

    # Log the number of poisoned samples that entered the clean pool (contamination check)
    logger.info(
        "{}/{} poisoned samples in the clean data pool".format(poison[total_indice].sum(), int(clean_pool_idx.sum()))
    )
    # Return the boolean mask representing the updated clean pool Dc
    return clean_pool_idx


class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
        labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
            set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)[0]
        self.labeled = labeled
        # self.prefetch = self.dataset.prefetch
        # if self.prefetch:
            # self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if self.labeled:
            item = self.dataset[self.semi_indice[index]]
            item["labeled"] = True
        else:
            item1 = self.dataset[self.semi_indice[index]]
            item2 = self.dataset[self.semi_indice[index]]
            img1, img2 = item1.pop("img"), item2.pop("img")
            item1.update({"img1": img1, "img2": img2})
            item = item1
            item["labeled"] = False

        return item

    def __len__(self):
        return len(self.semi_indice)


def class_agnostic_loss_guided_split(record_list, ratio, logger):
    """
    Implements class-agnostic loss-guided splitting (Phase 2 of ASD).
    Selects a fixed proportion (`ratio` or epsilon) of samples with the 
    lowest loss from the *entire* dataset, without considering their class labels.

    Args:
        record_list (list): List of metrics (must include 'loss' and 'poison').
        ratio (float): Proportion of the total dataset to be selected based on the lowest loss.
        logger: Logger instance.

    Returns:
        np.array: A boolean mask indicating the clean pool (Dc).
                  1 if a sample is in Dc, otherwise 0.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy() # Extract loss values
    poison = record_list[keys.index("poison")].data.numpy() # Extract poisoning flags
    clean_pool_idx = np.zeros(len(loss)) # Initialize the mask

    # Find the indices corresponding to the lowest `ratio * len(loss)` losses
    indice = loss.argsort()[: int(len(loss) * ratio)]
    # Log contamination status
    logger.info(
        "{}/{} poisoned samples in the clean data pool".format(poison[indice].sum(), len(indice))
    )
    # Update the mask
    clean_pool_idx[indice] = 1

    # Return the boolean mask representing the clean pool Dc
    return clean_pool_idx


def class_agnostic_loss_guided_split(record_list, ratio, logger):
    """
    Implements class-agnostic loss-guided splitting (Phase 2 of ASD).
    Selects a fixed proportion (`ratio` or epsilon) of samples with the 
    lowest loss from the *entire* dataset, without considering their class labels.

    Args:
        record_list (list): List of metrics (must include 'loss' and 'poison').
        ratio (float): Proportion of the total dataset to be selected based on the lowest loss.
        logger: Logger instance.

    Returns:
        np.array: A boolean mask indicating the clean pool (Dc).
                  1 if a sample is in Dc, otherwise 0.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy() # Extract loss values
    poison = record_list[keys.index("poison")].data.numpy() # Extract poisoning flags
    clean_pool_idx = np.zeros(len(loss)) # Initialize the mask

    # Find the indices corresponding to the lowest `ratio * len(loss)` losses
    indice = loss.argsort()[: int(len(loss) * ratio)]
    # Log contamination status
    logger.info(
        "{}/{} poisoned samples in the clean data pool".format(poison[indice].sum(), len(indice))
    )
    # Update the mask
    clean_pool_idx[indice] = 1

    # Return the boolean mask representing the clean pool Dc
    return clean_pool_idx


def meta_split(record_list, meta_record_list, ratio, logger):
    """
    Implements meta-splitting (Phase 3 of ASD).
    Selects samples based on the *minimum loss reduction* between 
    the original model and a "virtual" model trained for one epoch 
    on the full dataset. The idea is that clean hard samples are more 
    difficult to learn, leading to smaller loss reduction.

    Args:
        record_list (list): Metrics from the original model (before virtual training).
        meta_record_list (list): Metrics from the virtual model (after virtual training).
        ratio (float): Proportion of the total dataset to be selected based on the lowest loss reduction.
        logger: Logger instance.

    Returns:
        np.array: A boolean mask indicating the clean pool (Dc).
                  1 if a sample is in Dc, otherwise 0.
    """
    keys = [r.name for r in record_list]
    # Extract loss from the original model
    loss = record_list[keys.index("loss")].data.numpy()
    # Extract loss from the (trained) virtual model
    meta_loss = meta_record_list[keys.index("loss")].data.numpy()
    # Extract poisoning flags (used only for logging contamination status)
    poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_idx = np.zeros(len(loss)) # Initialize the mask

    # Compute loss reduction for each sample
    # A smaller reduction means the sample was harder to learn for the virtual model step.
    loss_reduction = loss - meta_loss

    # Find indices corresponding to the *smallest* `ratio * len(loss)` loss reductions
    indice = loss_reduction.argsort()[: int(len(loss) * ratio)]
    # Log contamination status
    logger.info(
        "{}/{} poisoned samples in the clean data pool".format(poison[indice].sum(), len(indice))
    )
    # Update the mask
    clean_pool_idx[indice] = 1

    # Return the boolean mask representing the clean pool Dc
    return clean_pool_idx


def train_the_virtual_model(meta_virtual_model, poison_train_loader, meta_optimizer, meta_criterion, device):
    """
    Train the "virtual" model for one epoch on the entire poisoned training dataset using standard supervised learning.
    This is part of the meta-split phase.

    Args:
        meta_virtual_model: A deep copy of the main model to be trained.
        poison_train_loader: DataLoader for the *entire* poisoned training dataset.
        meta_optimizer: Optimizer configured for the virtual model parameters.
        meta_criterion: Loss function for supervised training (e.g., CrossEntropyLoss).
        device (int): GPU index to use.
    """
    # Set the virtual model to training mode
    meta_virtual_model.train()
    # Iterate over batches of the full poisoned training data
    for batch_idx, batch in enumerate(poison_train_loader):
        # Move data and target to the specified GPU
        data = batch["img"].to(device)
        target = batch["target"].to(device)

        # Zero the gradients of the virtual model's optimizer
        meta_optimizer.zero_grad()
        # Forward pass: obtain predictions from the virtual model
        output = meta_virtual_model(data)
        # Ensure the loss reduction mode is set to 'mean' for standard training
        meta_criterion.reduction = "mean"
        # Compute the loss between predictions and ground truth targets
        loss = meta_criterion(output, target)

        # Backward pass: compute gradients
        loss.backward()
        # Update the weights of the virtual model
        meta_optimizer.step()


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=None):
        self.name = name
        self.reset()

    def reset(self):
        self.batch_avg = 0
        self.total_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, avg, n=1):
        self.batch_avg = avg
        self.sum += avg * n
        self.count += n
        self.total_avg = self.sum / self.count

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]


def mixmatch_train(
    model, xloader, uloader, criterion, optimizer, epoch, logger, train_iteration, temperature, alpha, num_classes
):

    loss_meter = AverageMeter("loss")
    xloss_meter = AverageMeter("xloss")
    uloss_meter = AverageMeter("uloss")
    lambda_u_meter = AverageMeter("lambda_u")
    meter_list = [loss_meter, xloss_meter, uloss_meter, lambda_u_meter]

    xiter = iter(xloader)
    uiter = iter(uloader)

    model.train()
    gpu = next(model.parameters()).device
    start = time.time()
    for batch_idx in range(train_iteration):
        try:
            xbatch = next(xiter)
            xinput, xtarget = xbatch["img"], xbatch["target"]
        except:
            xiter = iter(xloader)
            xbatch = next(xiter)
            xinput, xtarget = xbatch["img"], xbatch["target"]

        try:
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        except:
            uiter = iter(uloader)
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, num_classes).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        xinput = xinput.cuda(gpu, non_blocking=True)
        xtarget = xtarget.cuda(gpu, non_blocking=True)
        uinput1 = uinput1.cuda(gpu, non_blocking=True)
        uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / temperature)
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()


        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)

        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / train_iteration,
        )
        loss = Lx + lambda_u * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()

        loss_meter.update(loss.item())
        xloss_meter.update(Lx.item())
        uloss_meter.update(Lu.item())
        lambda_u_meter.update(lambda_u)
        # tabulate_step_meter(batch_idx, train_iteration, 3, meter_list, logger)

    logger.info("MixMatch training summary:")
    # tabulate_epoch_meter(time.time() - start, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result, model


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


def select_optimized_parameters(model, model_name):
    if "ResNet" in model_name:
        params = [{'params': model.backbone.module.layer3.parameters()},
                 {'params': model.backbone.module.layer4.parameters()},
                 {'params': model.linear.parameters()}]
    else:
        params = None
        raise NotImplementedError
    return params


class ASD(Training):
    def __init__(self, args):
        super().__init__(args)

    def train(self, train_dataset, test_dataset, model, aux_dataset=None):
        logger = logging.getLogger(__name__)
        self.epochs = self.config['global'].get("epoch_third", 120)
        # --- Create DataLoaders ---
        merged_dataset = CustomConcatDataset([train_dataset, aux_dataset])
        merged_dataset.train = True
        # Create a corresponding label list: train_dataset corresponds to 1, aux_dataset corresponds to 0
        poison_train_idx = [1] * len(train_dataset) + [0] * len(aux_dataset)
        poison_train_data = PoisonLabelDataset(merged_dataset, train_dataset.transform, poison_train_idx)
        poison_train_loader = DataLoader(poison_train_data, batch_size=self.bs, shuffle=True, num_workers=16, pin_memory=True)
        poison_eval_loader = DataLoader(poison_train_data, batch_size=self.bs, shuffle=False, num_workers=16, pin_memory=True)
        clean_test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=16, pin_memory=True)
        # --- Loss functions (Criteria) ---
        split_criterion = SCELoss(alpha=0.1, beta=1, num_classes=self.args.num_classes).to(self.device)
        semi_criterion = MixMatchLoss(lambda_u=15, rampup_length=self.epochs).to(self.device)
        # --- Optimizer and scheduler ---
        extractor, classifier = get_last_layer_and_replace_with_identity(model)
        linear_model = LinearModel(extractor, classifier)
        optimizer = get_optim(linear_model.parameters(), self.optim, self.lr, self.momentum, self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # --- Initialize clean seed samples (Start of Phase 1) ---
        # This corresponds to the initial clean pool Dc in the ASD paper (Section 4, Figure 2, Phase 1)
        clean_data_info = {}  # Dictionary to store indices of known clean samples for each class
        all_data_info = {}  # Dictionary to store indices of *all* samples for each class (to be pruned)
        # Initialize a dictionary with empty lists for each class
        for i in range(self.args.num_classes):
            clean_data_info[str(i)] = []
            all_data_info[str(i)] = []
        # Iterate through the *poisoned* training dataset (which contains both clean and poisoned samples)
        # to identify clean samples and populate the dictionaries
        for idx, item in enumerate(poison_train_data):
            if item['poison'] == 0:  # Check if the sample is labeled as clean
                clean_data_info[str(item['target'])].append(idx)  # Add index to clean info
            all_data_info[str(item['target'])].append(idx)  # Add index to all info
        
        # Select initial seed samples for the clean pool (Dc)
        indice = []  # This list will store indices of the initial clean pool Dc
        seed_num = self.config['global'].get("seed_num", 10)
        for k, v in clean_data_info.items():  # Iterate through clean samples of each class
            # Randomly select 'seed_num' clean samples *without replacement* for this class
            choice_list = np.random.choice(v, replace=False, size=seed_num).tolist()
            indice.extend(choice_list)  # Add selected indices to the initial clean pool
            # Remove selected seed samples from 'all_data_info' of that class,
            # so they are not considered again during splitting.
            all_data_info[k] = [x for x in all_data_info[k] if x not in choice_list]
        # Convert the index list into a NumPy array
        indice = np.array(indice)  # `indice` now represents the initial clean set Dc

        # --- Main training loop ---
        choice_num = 0  # Counter for the number of selected samples per class in Phase 1
        epoch_first = self.config['global'].get("epoch_first", 60)
        epoch_second = self.config['global'].get("epoch_second", 90)
        epoch_third = self.config['global'].get("epoch_third", 120)
        t = self.config['global'].get("t", 5)
        n = self.config['global'].get("n", 10)
        epsilon = self.config['global'].get("epsilon", 0.5)
        for epoch in range(self.config['global'].get("epoch_third", 120)):
            logger.info(
                "===Epoch: {}/{}===".format(epoch + 1, self.config['global'].get("epoch_third", 120))
            )

            # --- Data Splitting Phase (Core logic of ASD) ---
            if epoch < epoch_first:
                # --- Phase 1: Warm-up - Class-aware loss-guided splitting ---
                # Record the loss of each sample in the training set using the current model and split_criterion
                record_list = poison_linear_record(
                    linear_model, poison_eval_loader, split_criterion
                )
                # Periodically increase the number of clean samples selected per class
                if epoch % t == 0 and epoch != 0:
                    choice_num += n

                logger.info("Mining clean data via class-aware loss-guided splitting...")
                # Perform class-aware splitting: Select 'choice_num' samples with the lowest loss *per class*
                # from the remaining candidates (`all_data_info`), adding them to the initial `indice`.
                split_idx = class_aware_loss_guided_split(record_list, indice, all_data_info, choice_num, logger)
                # `split_idx` is a boolean mask: samples in the clean pool (Dc) are 1, others are 0.
                # Create datasets for semi-supervised learning:
                xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)  # Labeled clean pool Dc
                udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)  # Unlabeled poisoned pool Dp

            elif epoch < epoch_second:
                # --- Phase 2: Class-agnostic loss-guided splitting ---
                # Record losses as in Phase 1
                record_list = poison_linear_record(
                    linear_model, poison_eval_loader, split_criterion
                )
                logger.info("Mining clean data via class-agnostic loss-guided splitting...")
                # Perform class-agnostic splitting: Select a fixed proportion ('epsilon') of the lowest-loss samples
                # from the *entire* dataset, ignoring class labels.
                split_idx = class_agnostic_loss_guided_split(record_list, epsilon, logger)
                # `split_idx` is the updated boolean mask for clean pool Dc.
                # Create datasets:
                xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)  # Labeled clean pool Dc
                udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)  # Unlabeled poisoned pool Dp

            elif epoch < epoch_third:
                # --- Phase 3: Meta-Split ---
                # Record losses on the current main model
                record_list = poison_linear_record(
                    linear_model, poison_eval_loader, split_criterion
                )
                # Create a deep copy of the current model as the "virtual model"
                meta_virtual_model = deepcopy(linear_model)
                # Select parameters to optimize in the virtual model (e.g., Layer 3, Layer 4, and linear layer)
                param_meta = select_optimized_parameters(meta_virtual_model, self.args.model)
                # Instantiate optimizer for the virtual model
                meta_optimizer = torch.optim.Adam(param_meta, lr=self.config["meta"].get("lr", 0.015))

                # Get the loss function for training the virtual model (typically standard CE loss)
                meta_criterion = torch.nn.CrossEntropyLoss()

                # Train the virtual model for a few epochs (usually 1) on *the entire* poisoned dataset
                meta_epoch = self.config["meta"].get("epoch", 1)
                for _ in range(meta_epoch):
                    train_the_virtual_model(
                                            meta_virtual_model=meta_virtual_model,
                                            poison_train_loader=poison_train_loader,  # Use full poisoned dataset
                                            meta_optimizer=meta_optimizer,
                                            meta_criterion=meta_criterion,
                                            device=self.device
                                            )
                # Record losses on the *trained* virtual model
                meta_record_list = poison_linear_record(
                    meta_virtual_model, poison_eval_loader, split_criterion  # Use split_criterion for comparison
                )

                logger.info("Mining clean data via meta-split...")
                # Perform meta-split: Select samples based on the *loss reduction* between the original and virtual models.
                # Samples with the smallest reduction (harder to learn) are considered as potential clean hard samples.
                split_idx = meta_split(record_list, meta_record_list, epsilon, logger)
                # `split_idx` is the updated boolean mask for clean pool Dc.
                # Create datasets:
                xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)  # Labeled clean pool Dc
                udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)  # Unlabeled poisoned pool Dp


            # --- Semi-supervised training step ---
            # Create DataLoaders (Dc and Dp) for the current split
            # Use drop_last=True for MixMatch compatibility if the batch size does not align perfectly
            semi_batch_size = self.config["semi"].get("batch_size", 64)
            xloader = DataLoader(
                xdata, batch_size=semi_batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True
            ) # Labeled clean data loader (Dc)
            uloader = DataLoader(
                udata, batch_size=semi_batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True
            ) # Unlabeled contaminated data loader (Dp)

            logger.info("MixMatch training...")
            # Perform one epoch of semi-supervised training (e.g., MixMatch)
            # This trains the main `linear_model` using labeled (xloader) and unlabeled (uloader) data.
            poison_train_result, linear_model = mixmatch_train(
                linear_model, # Main model to be trained
                xloader,      # Labeled data loader
                uloader,      # Unlabeled data loader
                semi_criterion, # Loss function for semi-supervised learning
                optimizer,      # Optimizer for the main model
                epoch,          # Current epoch number
                logger,         # Logger instance
                **self.config["semi"]["mixmatch"] # Other MixMatch-specific parameters
            )

            scheduler.step()

        return linear_model
    