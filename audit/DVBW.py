#!~/Yanhengrui/anaconda3/envs/DVBW/bin python3
# -*- coding: utf-8 -*-

import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import random
import os
from PIL import Image
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from audit.dataset_audit import DatasetAudit
from utils.datasets import get_full_dataset

# Define the trigger appending transformation
class TriggerAppending(object):
    """
    Args:
         trigger: The trigger pattern (tensor with values in [0,1])
         alpha: The blending coefficient (tensor with values in [0,1])
         x_poisoned = (1-alpha)*x_benign + alpha*trigger
    """
    def __init__(self, trigger, alpha):
        self.trigger = np.array(trigger.clone().detach().permute(1, 2, 0) * 255)  # Convert to [0,255]
        self.alpha = np.array(alpha.clone().detach().permute(1, 2, 0))

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Image blended with the trigger pattern.
        """
        img_ = np.array(img).copy()
        img_ = (1 - self.alpha) * img_ + self.alpha * self.trigger
        return Image.fromarray(img_.astype('uint8')).convert('RGB')


class DVBW(DatasetAudit):
    """
    A class for dataset auditing, including watermark embedding and verification.
    """
    def __init__(self, args):
        self.image_size = args.image_size
        self.config = args.audit_config
        self.device = args.device
        self.reprocessing = args.reprocessing
        self.batch_size = args.bs
        self.dataset = args.dataset


    def process_dataset(self, ori_dataset, aux_dataset=None, params=None):
        """
        Embed a watermark into the original dataset without changing its format.

        Args:
            ori_dataset: The original dataset (in ImageFolder format).
            params (dict): Additional parameters for processing. Expected keys:
                - 'trigger': The watermark trigger.
                - 'alpha': The blending coefficient.
                - 'poison_rate': The rate of poisoned samples.
                - 'y_target': The target label for poisoned samples.

        Returns:
            tuple: (Processed dataset in ImageFolder format, auxiliary data for verification)
        """
        wm_dataset = ori_dataset
        wm_data_path = self.config.get("wm_data_path")
        if self.reprocessing and os.path.exists(wm_data_path):
            shutil.rmtree(wm_data_path)
        # Retrieve parameters
        if params is None:
            params = {}
        trigger = params.get('trigger', None)
        # alpha = params.get('alpha', None)
        poison_rate = self.config.get('poisoned_rate', 0.1)
        y_target = params.get('y_target', 1)
        alpha = self.config.get("alpha", 0.2)

        # Initialize trigger and alpha
        if trigger is None:
            trigger = torch.zeros([3, self.image_size, self.image_size], dtype=torch.float)
            # trigger[:, 29:32, 29:32] = 1  # White square watermark at the bottom-right corner
            for i in range(self.image_size):
                trigger[:, i, range(i % 2, self.image_size, 2)] = 1
                trigger[:, i, range((i + 1) % 2, self.image_size, 2)] = 0
            vutils.save_image(trigger.clone().detach(), 'Trigger_square.png')
        else:
            print('==> Loading the Trigger')
            trigger = Image.open(trigger)
            trigger = transforms.ToTensor()(trigger)
            assert torch.max(trigger) < 1.001, "Trigger is not normalized correctly."

        # if alpha is None:
        #     alpha = torch.ones([3, self.image_size, self.image_size], dtype=torch.float) * self.attack_config.get("alpha", 0.2)
        #     # alpha[:, 29:32, 29:32] = 1  # Full opacity at the bottom-right corner
        #     vutils.save_image(alpha.clone().detach(), 'Alpha_square.png')
        # else:
        #     print('==> Loading the Alpha')
        #     alpha = Image.open(alpha)
        #     alpha = transforms.ToTensor()(alpha)
        #     assert torch.max(alpha) < 1.001, "Alpha is not normalized correctly."

        # Resize trigger to the image size
        # trigger = transforms.Resize(self.image_size)(trigger)
        # alpha = transforms.Resize(self.image_size)(alpha)
        # Modify dataset samples in-place while keeping ImageFolder format
        poisoned_indices = random.sample(range(len(wm_dataset.samples)), int(len(wm_dataset.samples) * poison_rate))
        if self.reprocessing:
            classes = wm_dataset.classes
            for i, (img_path, label) in enumerate(wm_dataset.samples):
                img = Image.open(img_path).convert("RGB")
                img = transforms.ToTensor()(img)
                if i in poisoned_indices:
                    img = (1 - alpha) * img + alpha * trigger  # Apply trigger
                    label = y_target  # Change label to target class
                img = transforms.ToPILImage()(img.clamp(0, 1))  # Convert back to PIL image
                ff_path = os.path.join(wm_data_path, classes[label])
                if not os.path.exists(ff_path):
                    os.makedirs(ff_path)
                img_path = os.path.join(wm_data_path, classes[label], str(i) + ".png")
                img.save(img_path)  # Save the image with the poisoned one
                wm_dataset.samples[i] = (img_path, label)

            # Auxiliary data for verification
            aux = {'trigger': trigger, 'alpha': alpha, 'poisoned_indices': poisoned_indices}
        
        wm_dataset = datasets.ImageFolder(
            wm_data_path,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
        ])
        )
        aux = {'trigger': trigger, 'alpha': alpha, 'poisoned_indices': poisoned_indices, 'target_label': y_target}

        return wm_dataset, aux

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None, params: dict = None) -> dict:
        """
        Perform dataset auditing on the suspicious model and output the p-values of the T-test and Wilcoxon test.

        Note: The original verification code does not require the test dataset to be passed in;
        it loads a clean test dataset from './data/cifar10-imagefolder/test/'.
        To avoid bias from using the training set (pub_dataset), we load the test set directly
        and construct a watermarked version for comparison.

        Args:
            pub_dataset: Original dataset (ImageFolder format), not used directly.
            model: The model to be audited (already loaded).
            aux (dict): Auxiliary data required for verification, must contain {'trigger': trigger, 'alpha': alpha}.
            params (dict): Additional parameters, including:
                - 'target_label': Target label (default 1)
                - 'test_batch': Test batch size (default 128)
                - 'img_size': Image size (default 32)
                - 'margin': Margin added in the T-test (default 0.2)

        Returns:
            dict: A dictionary with keys {"ttest": T-test p-value, "wtest": Wilcoxon test confidence value (1-p)}.
        """
        # Extract the parameters from the auxiliary data
        # trigger_path = aux.get('trigger')
        # alpha_path = aux.get('alpha')
        # poisoned_idx = aux.get('poisoned_indices')
        # benign_idx = aux.get('benign_idx')

        # Ensure parameters dictionary exists
        if params is None:
            params = {}
        # Extract parameters from params (for target label, batch size, etc.)
        # target_label = params.get('target_label', 1)  # Default to target label 1
        # batch_size = params.get('test_batch', 512)  # Default batch size is 128
        # img_size = params.get('img_size', 32)  # Default image size is 32
        target_label = aux.get('target_label')

        # Load the clean dataset (ImageFolder format)
        # clean_dataset = get_full_dataset(self.dataset, (self.image_size, self.image_size))[1]
        clean_dataset = aux_dataset
        class PoisonedImageFolder(ImageFolder):
            def __init__(self, root, pattern, alpha, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
                super().__init__(root, transform=transform, target_transform=target_transform, 
                                loader=loader, is_valid_file=is_valid_file)
                self.pattern = pattern
                self.alpha = alpha

            def __getitem__(self, index):
                path, target = self.samples[index]
                sample = Image.open(path).convert("RGB")
                if self.transform is not None:
                    sample = self.transform(sample)
                sample = (1 - self.alpha) * sample + self.alpha * self.pattern
                sample = sample.clamp(0, 1)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return sample, target

        poisoned_dataset = PoisonedImageFolder(
            root=clean_dataset.root,
            pattern = aux.get("trigger"),
            alpha = aux.get("alpha"),
            transform=clean_dataset.transform,
            target_transform=clean_dataset.target_transform
        )
        # poisoned_dataset = pub_dataset
        # Create DataLoader for both clean dataset and watermarked dataset (pub_dataset)
        clean_loader = DataLoader(clean_dataset, batch_size=self.batch_size, shuffle=False)
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.batch_size, shuffle=False)

        # Inference using the model to obtain softmax probabilities for the T-test
        output_clean = self.test(clean_loader, model, params)
        output_poisoned = self.test(poisoned_loader, model, params)

        # poutput_clean = np.array([output_clean[i] for i in poisoned_idx]) range
        # poutput_poisoned = np.array([output_poisoned[i] for i in range(len(clean_dataset))])
        # print(np.argmax(poutput_poisoned, axis=1))
        # calculate WSR
        wsr = np.sum(np.argmax(output_poisoned, axis=1) == target_label) / output_poisoned.shape[0]

        # Extract probabilities for the target label
        p_clean = np.array([output_clean[i, target_label] for i in range(len(clean_dataset))])
        p_poisoned = np.array([output_poisoned[i, target_label] for i in range(len(clean_dataset))])

        # Perform paired T-test: check if the watermark significantly increases the target label probability
        margin = self.config.get("margin")
        _, p_ttest = ttest_rel(p_clean + margin, p_poisoned, alternative='less')

        # For the Wilcoxon test, use the model's predicted labels (argmax)
        # output_poisoned_labels = self.test_labels(poisoned_loader, model, params)
        # Calculate the difference between predicted labels and the target label
        # diff = output_poisoned_labels - target_label
        # try:
            # _, p_w = wilcoxon(diff, zero_method='zsplit', alternative='two-sided', mode='approx')
            # wtest_val = 1 - p_w
        # except Exception as e:
            # wtest_val = None

        # return {"ttest": p_ttest, "wtest": wtest_val}
        return {"p-value": p_ttest, "wsr": wsr}

    def test(self, testloader, model, params):
        """
        Perform inference on the test set using the model and return the softmax probabilities for each sample.
        """
        model.eval()
        return_output = []
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                return_output += torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy().tolist()
        return np.array(return_output)

    def test_labels(self, testloader, model, params):
        """
        Perform inference on the test set using the model and return the predicted labels (argmax) for each sample.
        """
        model.eval()
        predictions = []
        for _, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().detach().numpy().tolist())
        return np.array(predictions)



def get_dataset_auditing(args):
    if args.audit_method == "DVBW":
        return DVBW()
