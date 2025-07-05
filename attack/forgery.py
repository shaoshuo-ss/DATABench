import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import copy
import torch.nn as nn
from utils.models import get_model
from tqdm import tqdm
from utils.test import test_img
from utils.models import get_model
import torchattacks
import os
import logging


class VNIFGSMAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, device='cuda', alpha=0.01, steps=10,
                 target_label=None, transform=None, target_transform=None, bs=256, reverse=False):
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.device = device
        self.alpha = alpha
        self.steps = steps
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.reverse = reverse
        self.bs = bs
        self.perturbed_images = []
        self.targets = []

        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True)
        atk = torchattacks.VNIFGSM(model, eps=epsilon, alpha=alpha, steps=steps)
        if self.target_label is not None or reverse:
            atk.set_mode_targeted_by_label(quiet=True)
        for image, true_label in tqdm(dataloader):
            if self.target_label is None and not reverse:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            elif self.target_label is not None:
                tar_label = torch.full_like(true_label, self.target_label).to(self.device)
                adv_images = atk(image, tar_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            else:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class TIFGSMAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, device='cuda', alpha=0.01, steps=10,
                 target_label=None, transform=None, target_transform=None, bs=256, reverse=False):
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.device = device
        self.alpha = alpha
        self.steps = steps
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.reverse = reverse
        self.bs = bs
        self.perturbed_images = []
        self.targets = []

        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True)
        atk = torchattacks.TIFGSM(model, eps=epsilon, alpha=alpha, steps=steps)
        if self.target_label is not None or reverse:
            atk.set_mode_targeted_by_label(quiet=True)
        for image, true_label in tqdm(dataloader):
            if self.target_label is None and not reverse:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            elif self.target_label is not None:
                tar_label = torch.full_like(true_label, self.target_label).to(self.device)
                adv_images = atk(image, tar_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            else:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class LGVAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, device='cuda', alpha=0.01, steps=10,
                 target_label=None, transform=None, target_transform=None, bs=256, reverse=False):
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.device = device
        self.alpha = alpha
        self.steps = steps
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.reverse = reverse
        self.bs = bs
        self.perturbed_images = []
        self.targets = []

        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True)
        atk = torchattacks.LGV(model, dataloader, eps=epsilon, lr=alpha, epochs=steps, attack_class=torchattacks.PGD)
        if self.target_label is not None or reverse:
            atk.set_mode_targeted_by_label(quiet=True)
        for image, true_label in tqdm(dataloader):
            if self.target_label is None and not reverse:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            elif self.target_label is not None:
                tar_label = torch.full_like(true_label, self.target_label).to(self.device)
                adv_images = atk(image, tar_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
            else:
                adv_images = atk(image, true_label)
                self.perturbed_images.extend(list(torch.unbind(adv_images, dim=0)))
                self.targets.extend(true_label.tolist())
    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class FGSMAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, device='cuda',
                 target_label=None, transform=None, target_transform=None, bs=256, reverse=False):
        """
        original_dataset: torchvision.datasets.ImageFolder
        model: torch.nn.Module (should be in eval mode)
        epsilon: float, FGSM perturbation strength
        target_label: int or None
            - If None: untargeted attack
            - If int: targeted attack to this label
        transform: optional transform applied to perturbed images
        target_transform: optional transform applied to labels
        """
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.device = device
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.reverse = reverse

        self.perturbed_images = []
        self.targets = []
        self.bs = bs

        self._generate_adversarial_examples()

    def _generate_adversarial_examples(self):
        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False)
        # for i in range(len(self.dataset)):
            # image, true_label = self.dataset[i]
        for image, true_label in dataloader:

            # image: Tensor (already transformed by original_dataset)
            image_tensor = image.to(self.device)
            image_tensor.requires_grad = True

            # set label
            if self.target_label is None:
                label_tensor = true_label.to(self.device)
            else:
                label_tensor = (torch.ones_like(true_label) * self.target_label).long().to(self.device)

            # forward + backward
            output = self.model(image_tensor)
            loss = F.cross_entropy(output, label_tensor)
            self.model.zero_grad()
            loss.backward()

            data_grad = image_tensor.grad.data

            if self.reverse:
                data_grad = -1 * data_grad
            if self.target_label is None:
                # Untargeted: maximize loss
                perturbed_image = image_tensor + self.epsilon * data_grad.sign()
            else:
                # Targeted: minimize loss toward target
                perturbed_image = image_tensor - self.epsilon * data_grad.sign()

            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().cpu()

            self.perturbed_images.extend(list(torch.unbind(perturbed_image, dim=0)))
            self.targets.extend(true_label.tolist())

    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
class UAPAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, alpha=0.01, device='cuda', target_label=None,
                 max_iter=10, transform=None, target_transform=None, bs=256, reverse=False):
        """
        original_dataset: Original dataset (e.g., ImageFolder)
        model: Classification model (in eval mode)
        epsilon: Maximum allowed perturbation magnitude
        max_iter: Number of iterations to update the universal perturbation
        """
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.bs = bs
        self.target_label = target_label
        self.reverse = reverse
        self.alpha = alpha

        self.universal_perturbation = self._generate_uap()
        self.perturbed_images = [torch.clamp(x[0] + self.universal_perturbation, 0, 1) for x in self.dataset]
        self.targets = [x[1] for x in self.dataset]

    def _generate_uap(self):
        # Initialize perturbation
        delta = torch.zeros_like(self.dataset[0][0]).to(self.device)
        # delta.requires_grad = True
        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=4)
        for iteration in range(self.max_iter):
            for x, y in tqdm(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                if self.target_label is not None:
                    y_target = torch.full_like(y, self.target_label).to(self.device)
                else:
                    y_target = y
                v_batch = delta.detach().clone().unsqueeze(0).repeat(x.size(0), 1, 1, 1)
                v_batch.requires_grad = True # Enable gradient calculation for this batch op
                x_perturbed = torch.clamp(x + v_batch, 0, 1)
                # x_perturbed.requires_grad = True

                output = self.model(x_perturbed)
                # pred = output.argmax(dim=1)

                loss = F.cross_entropy(output, y_target)
                # self.model.zero_grad()

                loss.backward()
                grad = v_batch.grad.data
                grad_batch_mean = grad.mean(dim=0, keepdim=False)
                if self.reverse:
                    grad_batch_mean = -1 * grad_batch_mean
                if self.target_label is not None:
                    delta = delta - self.alpha * grad_batch_mean.sign()
                else:
                    delta = delta + self.alpha * grad_batch_mean.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon).detach()
                v_batch.grad.zero_()
                # delta.grad.zero_()

        return delta.detach().cpu()

    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class PGDAttackDataset(Dataset):
    def __init__(self, original_dataset, model, epsilon=0.03, alpha=0.01, iters=10,
                 device='cuda', target_label=None, transform=None, target_transform=None, bs=256, reverse=False):
        """
        original_dataset: Original dataset
        model: Model (in eval mode)
        epsilon: Maximum perturbation range
        alpha: Perturbation strength per step
        iters: Number of iterations
        target_label: Target label (None for untargeted attack)
        bs: Batch size for processing
        """
        self.dataset = original_dataset
        self.model = model.to(device).eval()
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.device = device
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.bs = bs
        self.reverse = reverse

        self.perturbed_images = []
        self.targets = []

        self._generate_adversarial_examples()

    def _generate_adversarial_examples(self):
        dataloader = DataLoader(self.dataset, batch_size=self.bs, shuffle=False)
        for x_batch, y_batch in dataloader:
            x_orig = x_batch.clone().to(self.device)
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon).to(self.device)
            x_adv = x_adv.clamp(0, 1).detach().requires_grad_(True)

            if self.target_label is None:
                y_tensor = y_batch.to(self.device)
            else:
                y_tensor = (torch.ones_like(y_batch) * self.target_label).long().to(self.device)

            for _ in range(self.iters):
                output = self.model(x_adv)
                loss = F.cross_entropy(output, y_tensor)
                self.model.zero_grad()
                loss.backward()

                grad_sign = x_adv.grad.data.sign()
                if self.reverse:
                    grad_sign = -1 * grad_sign
                if self.target_label is None:
                    x_adv = x_adv + self.alpha * grad_sign
                else:
                    x_adv = x_adv - self.alpha * grad_sign

                # Project back to epsilon ball
                x_adv = torch.max(torch.min(x_adv, x_orig + self.epsilon), x_orig - self.epsilon)
                x_adv = x_adv.clamp(0, 1).detach().requires_grad_(True)

            self.perturbed_images.extend(list(torch.unbind(x_adv.detach().cpu(), dim=0)))
            self.targets.extend(y_batch.tolist())

    def __len__(self):
        return len(self.perturbed_images)

    def __getitem__(self, idx):
        image = self.perturbed_images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label



def get_forged_dataset(args, dataset, model, attack_method):
    config = args.audit_config
    if attack_method == "fgsm":
        attack_dataset = FGSMAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["fgsm"].get("epsilon", 0.03),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    elif attack_method == "pgd":
        attack_dataset = PGDAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["pgd"].get("epsilon", 0.03),
            alpha=config["pgd"].get("alpha", 0.01),
            iters=config["pgd"].get("iters", 10),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    elif attack_method == "uap":
        attack_dataset = UAPAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["uap"].get("epsilon", 0.03),
            alpha=config["uap"].get("alpha", 0.01),
            max_iter=config["uap"].get("max_iter", 10),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    elif attack_method == "lgv":
        attack_dataset = LGVAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["lgv"].get("epsilon", 0.03),
            alpha=config["lgv"].get("alpha", 0.01),
            steps=config["lgv"].get("steps", 10),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    elif attack_method == "tifgsm":
        attack_dataset = TIFGSMAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["tifgsm"].get("epsilon", 0.03),
            alpha=config["tifgsm"].get("alpha", 0.01),
            steps=config["tifgsm"].get("steps", 10),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    elif attack_method == "vnifgsm":
        attack_dataset = VNIFGSMAttackDataset(
            original_dataset=dataset,
            model=model,
            epsilon=config["vnifgsm"].get("epsilon", 0.03),
            alpha=config["vnifgsm"].get("alpha", 0.01),
            steps=config["vnifgsm"].get("steps", 10),
            device=args.device,
            target_label=config.get("target_label"),
            bs=args.bs,
            reverse=config.get("reverse", False),
            # transform=dataset.transform,
            # target_transform=dataset.target_transform
        )
    else:
        raise ValueError("Unsupported attack method: {}".format(attack_method))
    return attack_dataset


def train_student_with_distillation(teacher_model, student_model, dataset, config, device, test_dataset, args):
    """
    Train a student model using knowledge distillation.
    
    Parameters:
        teacher_model: Pre-trained teacher model (nn.Module)
        student_model: Student model to be trained (nn.Module)
        dataset: Dataset used for training (torch.utils.data.Dataset)
        config: Dictionary containing training hyperparameters
            - epochs: Number of training epochs
            - batch_size: Size of each batch
            - lr: Learning rate
            - alpha: Weight for ground truth loss (1-alpha is used for distillation loss)
            - temperature: Distillation temperature
            - device: 'cuda' or 'cpu'
    
    Returns:
        Trained student model
    """
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 64)
    lr = config.get('lr', 1e-3)
    alpha = config.get('alpha', 0.5)
    temperature = config.get('temperature', 4.0)
    logger = logging.getLogger(__name__)

    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()

    transform = transforms.Compose([
            dataset.transform,
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
    ])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    for epoch in tqdm(range(epochs)):
        student_model.train()
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)

            # loss: soft target + ground truth
            loss_ce = ce_loss(student_logits, labels)
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc_val, _ = test_img(student_model, test_dataset, args)

        logger.info(f"Epoch {epoch+1}/{epochs} | Validation Accuracy: {acc_val:.4f}")
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    return student_model


def get_auxiliary_model(args, model, dataset, test_dataset):
    config = args.audit_config
    ori_model = model
    args.model = config["black-box"]["model"]
    aux_model = get_model(args)
    black_box_config = config["black-box"]
    if not black_box_config.get("retrain") and os.path.exists(black_box_config.get("pre-trained-path")):
        aux_model.load_state_dict(torch.load(black_box_config.get("pre-trained-path"), map_location="cpu"))
    else:
        aux_model = train_student_with_distillation(
            teacher_model=ori_model,
            student_model=aux_model,
            dataset=dataset,
            config=config["black-box"],
            device=args.device,
            test_dataset=test_dataset,
            args=args
        )
        os.makedirs(os.path.dirname(black_box_config.get("pre-trained-path")), exist_ok=True)
        torch.save(aux_model.state_dict(), black_box_config.get("pre-trained-path"))
    aux_model = aux_model.to(args.device)
    return aux_model


