# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
import random
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
# from audit.model import ResNet18 as resnet18
import torchvision.transforms as transforms
from audit.utils import save_imagefolder
from scipy.stats import ttest_rel
import logging
import shutil
from audit.utils import save_images_by_label
from scipy.stats import ttest_rel
from utils.datasets import get_full_dataset
from utils.models import get_swinvit

class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.

    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud
        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)

class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta, t_label):
        self.dataset = dataset
        self.delta = delta
        self.t_label = t_label

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img + self.delta[idx], target)

    def __len__(self):
        return len(self.dataset)

class Deltaset_(torch.utils.data.Dataset):
    def __init__(self, dataset, delta, t_label):
        self.dataset = dataset
        self.delta = delta
        self.t_label = t_label

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img, img + self.delta[idx], target)

    def __len__(self):
        return len(self.dataset)


class UBWC:
    """
    A class for dataset auditing, including watermark embedding and verification.
    """

    def __init__(self, args):
        """
        Initialize the DatasetAudit class.

        Args:
            args: Configuration arguments containing dataset and attack parameters.
        """
        self.image_size = args.image_size
        self.config = args.audit_config
        self.device = args.device
        self.reprocessing = args.reprocessing
        self.batch_size = args.bs
        self.dataset = args.dataset
        self.save_path = args.save_path
        self.source_class = self.config.get("source_class", 2)  
        self.target_class = self.config.get("target_class", 3)  
        self.poison_num = self.config.get("poison_num", 3000)
        self.eps = self.config.get("eps", 16. / 255)
        self.patch_size = self.config.get("patch_size", 8)
        self.class_num = args.num_classes
        # please note that the craft_iters should be set to an integer multiple of the retrain_iters
        self.craft_iters = self.config.get("craft_iters", 250)      
        self.retrain_iters = self.config.get("retrain_iters", 50)   
        self.train_epochs = self.config.get("train_epochs", 40)
        self.wm_data_path = self.config.get("wm_data_path")
        self.beta = self.config.get("beta", 2.0) # for 'TinyImageNet' use 0.8
        self.init_model_path = self.config.get("init_model_path", None)  
        self.args = args
        
        if not os.path.exists(self.wm_data_path):
            os.makedirs(self.wm_data_path)   

        if self.dataset == 'cifar10-imagefolder' or self.dataset == 'GTSRB':
            params = dict(source_size=self.image_size, target_size=self.image_size, shift=8, fliplr=True)
            self.paugment = RandomTransform(**params, mode='bilinear')
            self.augment = RandomTransform(source_size=self.image_size, target_size=self.image_size, shift=8, fliplr=True, mode='bilinear')
        elif self.dataset == 'ImageNet' or self.dataset == 'imagenet100':
            params = dict(source_size=self.image_size, target_size=self.image_size, shift=64, fliplr=True)
            self.paugment = RandomTransform(**params, mode='bilinear')
            self.augment = RandomTransform(source_size=self.image_size, target_size=self.image_size, shift=64, fliplr=True, mode='bilinear')
        elif self.dataset == 'TinyImageNet':
            params = dict(source_size=self.image_size, target_size=self.image_size, shift=16, fliplr=True)
            self.paugment = RandomTransform(**params, mode='bilinear')
            self.augment = RandomTransform(source_size=self.image_size, target_size=self.image_size, shift=16, fliplr=True, mode='bilinear')

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Embed a watermark into the original dataset.

        Args:
            ori_dataset (ImageFolder): The original dataset.

        Returns:
            A tuple containing:
                - pub_dataset (ImageFolder): The processed dataset with embedded watermark.
                - aux (dict): Auxiliary data required for verification.
        """
        # Prepare the source dataset (source class samples)
        source_dataset = self._prepare_source_dataset(ori_dataset)

            # prepare trained model
        model = self._get_model()
        if os.path.exists(os.path.join("final/UBWC/noattack/",self.dataset, "init_model.pth")):
            model.load_state_dict(torch.load(os.path.join("final/UBWC/noattack/",self.dataset, "init_model.pth"), map_location="cpu"))  
        else:
            self._train_model(model, ori_dataset, init=True)
            
        # Prepare the poisoned dataset (select poison samples and generate poison deltas)
        poison_set, poison_ids, poison_lookup, poison_reverse_lookup = self._prepare_poisoned_dataset(model, ori_dataset)
            
        # Calculate target gradient and gradient norm 
        source_grad, source_grad_norm = self._get_gradient(model,
                                             torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=False,
                                                                         drop_last=False), nn.CrossEntropyLoss())
        if self.reprocessing:
            # Generate initial poison deltas
            poison_deltas = self._initialize_poison_deltas(self.poison_num, ori_dataset[0][0].shape, self.eps)
            # Joint optimization
            pub_dataset = self._joint_optimization(model, ori_dataset, poison_set, poison_ids, poison_reverse_lookup, source_dataset, poison_deltas, source_grad, source_grad_norm)

            # Save the poisoned dataset if reprocessing is enabled
            save_images_by_label(pub_dataset, self.wm_data_path, ori_dataset.classes)
        pub_dataset = ImageFolder(
            self.wm_data_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size)
            ])
        )
        # Auxiliary data for verification
        aux = {
            'poison_num': self.poison_num,
            'poison_ids': poison_ids,
            'poison_lookup': poison_lookup,
            'poison_reverse_lookup': poison_reverse_lookup
        }

        return pub_dataset, aux

    def _prepare_source_dataset(self, dataset, test_flag=False):
        """
        Prepare the source dataset by selecting samples of the source class.

        Args:
            dataset (ImageFolder): The original dataset.

        Returns:
            ImageFolder: A new dataset containing only samples of the source class.
        """
        source_dataset = ImageFolder(
            root=dataset.root,  
            transform=dataset.transform,
            target_transform=dataset.target_transform,
            loader=dataset.loader,
        )

        source_dataset.samples = [(path, label) for path, label in source_dataset.samples if label in self.source_class]
        source_dataset.imgs = source_dataset.samples  
        return self._patch_source(source_dataset, self.target_class, vflag=test_flag)

    def _patch_source(self, dataset, target_label, vflag=False, random_patch=True):
        """
        Patch the source dataset with a trigger.
        """
        trigger = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
        patch = trigger.repeat((3, 1, 1))
        resize = torchvision.transforms.Resize((self.patch_size))
        patch = resize(patch)
        source_delta = []
        for idx, (source_img, label) in enumerate(dataset):
            if random_patch:
                patch_x = random.randrange(0, source_img.shape[1] - patch.shape[1] + 1)
                patch_y = random.randrange(0, source_img.shape[2] - patch.shape[2] + 1)
            else:
                patch_x = source_img.shape[1] - patch.shape[1]
                patch_y = source_img.shape[2] - patch.shape[2]

            delta_slice = torch.zeros_like(source_img).squeeze(0)
            diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
            delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
            source_delta.append(delta_slice.cpu())
        if not vflag:
            return Deltaset(dataset, source_delta, target_label)
        else:
            return Deltaset_(dataset, source_delta, target_label)

    def _prepare_poisoned_dataset(self, model, dataset):
        """
        Prepare the poisoned dataset by selecting samples to be poisoned.

        Args:
            model: The model used for gradient calculation.
            dataset (ImageFolder): The dataset containing samples.

        Returns:
            tuple: A tuple containing the poisoned dataset, poison IDs, poison lookup, and poison reverse lookup.
        """
        poison_ids = self._select_poison_ids(model, dataset)
        poison_set = [dataset[i] for i in poison_ids]
        poison_lookup = dict(zip(range(len(poison_ids)), poison_ids))
        poison_reverse_lookup = dict(zip(poison_ids, range(len(poison_ids))))
        return poison_set, poison_ids, poison_lookup, poison_reverse_lookup

    def _select_poison_ids(self, model, dataset):
        """
        Select the IDs of samples to be poisoned based on gradient norms.

        Args:
            model: The model used for gradient calculation.
            dataset (ImageFolder): The dataset containing samples.

        Returns:
            list: A list of IDs of samples to be poisoned.
        """
        model.eval()
        grad_norms = []
        differentiable_params = [p for p in model.parameters() if p.requires_grad]
        tbar = tqdm(torch.utils.data.DataLoader(dataset))
        tbar.set_description('Calculating Gradients')
        for image, label in tbar:
            image, label = image.to(self.device), label.to(self.device)
            loss = F.cross_entropy(model(image), label)
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            grad_norms.append(grad_norm.sqrt().item())

        print('len(grad_norms):', len(grad_norms))
        poison_ids = np.argsort(grad_norms)[-self.poison_num:]
        return poison_ids

    def _joint_optimization(self, model, ori_dataset, poison_set, poison_ids, poison_reverse_lookup, source_dataset, poison_deltas, source_grad, source_grad_norm):
        """
        Joint optimization of poison deltas and model.

        Args:
            model: The model used for optimization.
            ori_dataset (ImageFolder): The original dataset.
            poison_set (list): The list of poisoned samples.
            poison_ids (list): The list of IDs of poisoned samples.
            poison_reverse_lookup (dict): A dictionary mapping poison IDs to their indices.
            source_dataset: The source dataset for gradient calculation.
            poison_deltas 
            target_grad
            target_gnorm

        Returns:
            The poisoned dataset after joint optimization.
        """
        model.eval()
        att_optimizer = torch.optim.Adam([poison_deltas], lr=0.025, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            att_optimizer, 
            milestones=[self.craft_iters // 2.667, self.craft_iters // 1.6, self.craft_iters // 1.142],
            gamma=0.1
        )
        # poison_deltas.requires_grad_(True)
        # poison_deltas.grad = torch.zeros_like(poison_deltas)
        if self.dataset == 'imagenet100':
            dataloader = torch.utils.data.DataLoader(poison_set, batch_size=32, drop_last=False, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(poison_set, batch_size=self.batch_size, drop_last=False, shuffle=False)

        for t in tqdm(range(1, self.craft_iters + 1), desc="Crafting Poison Deltas", unit="iter"):
            base = 0
            target_losses, poison_correct = 0., 0.
            poison_imgs = []
            model.eval()
            poison_deltas.grad = torch.zeros_like(poison_deltas)
            for imgs, targets in tqdm(dataloader, desc=f"Iteration {t}/{self.craft_iters}", unit="batch", leave=False):
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                loss, prediction = self._batched_step(model, imgs, targets, poison_deltas, list(range(base, base + len(imgs))), 
                                                      F.cross_entropy, source_grad, source_grad_norm)
                target_losses += loss
                poison_correct += prediction
                base += len(imgs)
                poison_imgs.append(imgs)

            poison_deltas.grad.sign_()
            att_optimizer.step() 
            scheduler.step()
            att_optimizer.zero_grad()
            
            with torch.no_grad():
                # Projection Step
                poison_imgs = torch.cat(poison_imgs)
                poison_deltas.data = torch.max(torch.min(poison_deltas, torch.ones_like(poison_deltas) * self.eps),
                                            -torch.ones_like(poison_deltas) * self.eps)
                poison_deltas.data = torch.max(torch.min(poison_deltas, 1 - poison_imgs), -poison_imgs)

            target_losses = target_losses / (len(dataloader) + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if t % 10 == 0:
                print(f'Iteration {t}: Target loss is {target_losses:2.4f}, '
                    f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if t % self.retrain_iters == 0 and t != self.craft_iters:
                temp_poison_trainset = self._generate_poisoned_dataset(ori_dataset, poison_deltas, poison_ids, poison_reverse_lookup)
                model = self._get_model()
                self._train_model(model, temp_poison_trainset)
                source_grad, source_grad_norm = self._get_gradient(model, torch.utils.data.DataLoader(source_dataset, batch_size=self.batch_size,
                                                                                        shuffle=False, drop_last=False),
                                                        nn.CrossEntropyLoss())
            
        return temp_poison_trainset
                
    def _generate_poisoned_dataset(self, ori_dataset, poison_deltas, poison_ids, poison_reverse_lookup):
        """
        Generate the poisoned dataset by applying the poison deltas to the original dataset.

        Args:
            ori_dataset (ImageFolder): The original dataset.
            poison_ids (list): The list of IDs of poisoned samples.
            poison_reverse_lookup (dict): A dictionary mapping poison IDs to their indices.
            poison_deltas (torch.Tensor):

        Returns:
            The poisoned dataset as a list of (image, label) tuples.
        """
        poisoned_trainset = []
        for i in range(len(ori_dataset)):
            if i not in poison_ids:
                poisoned_trainset.append(ori_dataset[i])
            else:
                poisoned_trainset.append((ori_dataset[i][0] + poison_deltas[poison_reverse_lookup[i]].cpu(), ori_dataset[i][1]))
        return poisoned_trainset

    def _initialize_poison_deltas(self, num_poison_deltas, input_shape, eps):
        """
        Initialize the poison deltas.

        Args:
            num_poison_deltas (int): The number of poison deltas to initialize.
            input_shape (tuple): The shape of the input images.

        Returns:
            torch.Tensor: The initialized poison deltas.
        """
        poison_deltas = ((torch.rand(num_poison_deltas, *input_shape) - 0.5) * 2).to(self.device)
        poison_deltas = (poison_deltas * eps).to(self.device)
        return poison_deltas

    def _get_model(self):
        """
        Get the model used for gradient calculation.

        Returns:
            torch.nn.Module: The model.
        """
        model = resnet18()
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=self.class_num)
        if self.class_num <= 10:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        def replace_bn_with_gn(model, num_groups=32):
            for name, module in model.named_children():
                if isinstance(module, nn.BatchNorm2d):
                    num_channels = module.num_features
                    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                    setattr(model, name, gn)
                else:
                    replace_bn_with_gn(module, num_groups)
        replace_bn_with_gn(model)
        model = model.to(self.device)
        return model
    
    def _get_gradient(self, model, train_loader, criterion):
        """
        Calculate the target gradient and target gradient norm.

        Args:
            source_dataset (ImageFolder): The source dataset containing only samples of the source class.

        Returns:
            A tuple (target_grad, target_gnorm).
        """
        model.eval()
        eps= 1e-12
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            outputs = F.softmax(outputs, dim=1)
            D_loss = outputs * (outputs + eps).log()
            D_loss = -self.beta * D_loss.sum(1)          #  entropy term
            loss += D_loss.sum() / len(labels)
            loss = -loss                            # for gradient ascending
            if batch_idx == 0:
                gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
            else:
                gradients = tuple(
                    map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, model.parameters(), only_inputs=True)))
        gradients = tuple(map(lambda i: i / len(train_loader.dataset), gradients))

        grad_norm = 0
        for grad_ in gradients:
            grad_norm += grad_.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, target_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # default self.args.centreg is 0, self.retain is False from the gradient matching repo
            global passenger_loss
            outputs = model(inputs)
            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
            passenger_loss = self._get_passenger_loss(poison_grad, target_grad, target_gnorm)
            passenger_loss.backward(retain_graph=False)
            return passenger_loss.detach(), prediction.detach()

        return closure

    def _get_passenger_loss(self, poison_grad, target_grad, target_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0
        indices = torch.arange(len(target_grad))
        for i in indices:
            passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
            poison_norm += poison_grad[i].pow(2).sum()

        passenger_loss = passenger_loss / target_gnorm  # this is a constant
        passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        return passenger_loss

    def _batched_step(self, model, inputs, labels, poison_deltas, poison_slices, criterion, target_grad, target_gnorm):
        """
        Take a step toward minimizing the current target loss.

        Args:
            model: The model to be audited.
            inputs: The input images.
            labels: The labels of the input images.
            poison_slices: The indices of the poison deltas for the current batch.
            poison_deltas (torch.Tensor): 
            target_grad, target_gnorm: 

        Returns:
            A tuple containing:
                - loss: The loss value.
                - prediction: The prediction accuracy.
        """
        delta_slice = poison_deltas[poison_slices]
        delta_slice.requires_grad_(True)
        poisoned_inputs = inputs.detach() + delta_slice
        closure = self._define_objective(self.paugment(poisoned_inputs), labels)
        loss, prediction = closure(model, criterion, target_grad, target_gnorm)
        poison_deltas.grad[poison_slices] = delta_slice.grad.detach()
        return loss.item(), prediction.item()

    def _train_model(self, model, trainset, init=False):
        """
        Train the model on the poisoned dataset.

        Args:
            model: The model to be trained.
            trainset: The training dataset.
        """
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=8
        )
        testset = get_full_dataset(self.dataset, (self.image_size, self.image_size))[1]
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        
        source_testset = self._prepare_source_dataset(testset)
        poison_sourceloader = torch.utils.data.DataLoader(source_testset, batch_size=self.batch_size, num_workers=8)
        
        full_patch_testset = self._patch_source(testset, self.target_class)
        poison_testloader = torch.utils.data.DataLoader(full_patch_testset, batch_size=self.batch_size, num_workers=8)
        
        if init == True:
            if self.dataset == 'cifar10-imagefolder' or self.dataset == 'GTSRB':
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
            elif self.dataset == 'TinyImageNet':
                epochs = 15
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            elif self.dataset == 'imagenet100':
                epochs = 150
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        else:
            if self.dataset == 'cifar10-imagefolder' or self.dataset == 'imagenet100':
                epochs = self.train_epochs
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            elif self.dataset == 'imagenet100':
                pass        # for finetuning use
                epochs = 60
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
            
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75])
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0.0

            with tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', unit='batch') as tepoch:
                for img, y in tepoch:
                    with torch.no_grad():
                        img, y = self.augment(img.to(self.device)), y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(img)
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    corrects = torch.sum(preds == y).item()
                    train_correct += corrects
                    train_loss += loss.item() * len(y)

                    tepoch.set_postfix(loss=loss.item(), accuracy=corrects / len(y))

            scheduler.step()
            train_loss, train_correct = train_loss / len(trainset), train_correct * 100. / len(trainset)
            model.eval()
            with torch.no_grad():
                test_correct = 0
                test_loss = 0
                for img, y in testloader:
                    img, y = img.to(self.device), y.to(self.device)
                    outputs = model(img)
                    loss = F.cross_entropy(outputs, y)
                    test_loss += loss.item() * len(y)
                    test_correct += (outputs.max(1)[1] == y).sum().item()
                test_loss, test_correct = test_loss / len(testset), test_correct * 100. / len(testset)
                ps_correct = 0
                ps_loss = 0
                for img, y in poison_sourceloader:
                    img, y = img.to(self.device), y.to(self.device)
                    outputs = model(img)
                    loss = F.cross_entropy(outputs, y)
                    ps_loss += loss.item() * len(y)
                    ps_correct += (outputs.max(1)[1] != y).sum().item()
                ps_loss, ps_correct = ps_loss / len(source_testset), ps_correct * 100. / len(source_testset)
                pt_correct = 0
                pt_loss = 0
                for img, y in poison_testloader:
                    img, y = img.to(self.device), y.to(self.device)
                    outputs = model(img)
                    loss = F.cross_entropy(outputs, y)
                    pt_loss += loss.item() * len(y)
                    pt_correct += (outputs.max(1)[1] != y).sum().item()
                pt_loss, pt_correct = pt_loss / len(full_patch_testset), pt_correct * 100. / len(full_patch_testset)
            
            if not init:
                logger = logging.getLogger(__name__)
                logger.info(
                    "epoch:%d, tr_loss:%.4f, tr_acc%.4f, te_loss:%.4f, te_acc:%.4f, psrc_loss:%.4f, psrc_acc:%.4f, pte_loss:%.4f, pte_acc:%.4f" % \
                    (epoch, train_loss, train_correct, test_loss, test_correct, ps_loss, ps_correct, pt_loss, pt_correct)
                )
            
        if init == True:
            torch.save(model.state_dict(), os.path.join("final/UBWC/noattack/",self.dataset, "init_model.pth"))

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None) -> float:
        """
        Verify the suspicious model by evaluating the effect of the patch.
        """
        model.eval()

        # testset = get_full_dataset(self.dataset, (self.image_size, self.image_size))[1]
        testset = aux_dataset
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        
        source_testset = self._prepare_source_dataset(testset, test_flag=True)
        patch_test_loader = torch.utils.data.DataLoader(source_testset, batch_size=128, num_workers=8)
        
        full_patch_testset = self._patch_source(testset, self.target_class, vflag=True)
        full_patch_test_loader = torch.utils.data.DataLoader(full_patch_testset, batch_size=128, num_workers=8)
        
        def calculate_metrics(data_loader):
            running_corrects = 0.0
            p_running_corrects = 0.0
            metric = np.zeros((self.class_num, self.class_num))

            with torch.no_grad():
                for idx, (inputs, p_inputs, labels) in enumerate(data_loader):
                    inputs = inputs.to(self.device)
                    p_inputs = p_inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    p_outputs = model(p_inputs)
                    _, p_preds = torch.max(p_outputs, 1)
                    
                    corrects = (preds == labels.data)
                    running_corrects += torch.sum(corrects)
                    p_corrects = (preds == labels.data) * (p_preds != labels.data)
                    p_running_corrects += torch.sum(p_corrects)

                    for label, pred in zip(labels, p_preds):
                        metric[label, pred] += 1

            poison_accuracy = p_running_corrects / running_corrects if running_corrects > 0 else 0.0
            clean_accuracy = running_corrects / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0

            return poison_accuracy, clean_accuracy, metric

        poison_accuracy, clean_accuracy, metric = calculate_metrics(patch_test_loader)
        _, full_clean_accuracy, _ = calculate_metrics(full_patch_test_loader)
        logger = logging.getLogger(__name__)
        logger.info('Poison Accuracy (ASR): {}'.format(poison_accuracy))
        logger.info('Clean Accuracy : {}'.format(full_clean_accuracy))
        logger.info('Class-wise Transfer Metric:\n{}'.format(metric))

        labels = []
        for img, label in testset:
            labels.append(label)
        labels = np.array(labels)
        
        output_clean = self.test(testloader, model, None)
        
        full_patch_testset = self._patch_source(testset, self.target_class)
        full_patch_test_loader = torch.utils.data.DataLoader(full_patch_testset, batch_size=128, num_workers=8)
        output_poisoned = self.test(full_patch_test_loader, model, None)

        source_class_indices = np.where(labels == self.source_class)[0]
        output_clean_source = output_clean[source_class_indices]
        output_poisoned_source = output_poisoned[source_class_indices]
        labels_source = labels[source_class_indices]

        wsr = np.sum(np.argmax(output_poisoned_source, axis=1) != labels_source) / output_poisoned_source.shape[0]

        p_clean = np.array([output_clean_source[i, labels_source[i]] for i in range(len(source_class_indices))])
        p_poisoned = np.array([output_poisoned_source[i, labels_source[i]] for i in range(len(source_class_indices))])

        margin = self.config.get("margin", 0.2)
        _, p_ttest = ttest_rel(p_poisoned + margin, p_clean, alternative='less')

        logger.info('WSR (Source Class): {}'.format(wsr))
        logger.info('p-value: {}'.format(p_ttest))

        return {"p-value": p_ttest, "wsr": wsr}
    
    def test(self, testloader, model, params):
        """
        Perform inference on the test set using the model and return the softmax probabilities for each sample.
        """
        model.eval()
        return_output = []
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                return_output += torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy().tolist()
        return np.array(return_output)

