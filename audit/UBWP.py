# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from PIL import Image
import copy
import torchvision.transforms.functional as F
from audit.utils import save_imagefolder
from torchvision.datasets.folder import default_loader
import logging
from scipy.stats import ttest_rel

from utils.datasets import get_full_dataset

# class AddTrigger:
    # def __init__(self):
        # pass


class AddTrigger:
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddTrigger, self).__init__()

        # Accelerated calculation
        self.weight = weight
        self.pattern = pattern
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        # img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img
    
    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return self.weight * img + self.res

class UBWP:
    """
    A class for dataset auditing, including watermark embedding and verification.
    """

    def __init__(self, args):
        self.image_size = args.image_size
        self.config = args.audit_config
        self.device = args.device
        self.reprocessing = args.reprocessing
        self.num_classes = args.num_classes
        self.batch_size = args.bs
        self.dataset = args.dataset

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Embed a watermark into the original dataset.

        Args:
            ori_dataset (ImageFolder): The original dataset.
            poisoned_rate (float): Ratio of poisoned samples.
            pattern (torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W). 
                If None, a default CIFAR10-style pattern will be used.
            weight (torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W). 
                If None, a default CIFAR10-style weight will be used.
            poisoned_transform_index (int): The position index that poisoned transform will be inserted.
            num_class (int): Number of classes in the dataset.

        Returns:
            (pub_dataset, aux): Processed dataset and auxiliary info.
        """
        poisoned_rate = self.config.get("poisoned_rate", 0.1)
        weight = self.config.get("alpha", 0.2)
        pattern = None
        # weight = None
        poisoned_transform_index = -1
        num_class = self.num_classes
        total_num = len(ori_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        poisoned_idx = random.sample(range(total_num), poisoned_num)

        if pattern is None:
            pattern = torch.zeros([3, self.image_size, self.image_size], dtype=torch.float)
            # trigger[:, 29:32, 29:32] = 1  # White square watermark at the bottom-right corner
            for i in range(self.image_size):
                pattern[:, i, range(i % 2, self.image_size, 2)] = 1
                pattern[:, i, range((i + 1) % 2, self.image_size, 2)] = 0
        # if weight is None:
            # weight = torch.zeros((3, 32, 32), dtype=torch.float32)
            # weight[:, -3:, -3:] = 1.0
            # weight = F.resize(weight, [self.image_size, self.image_size])

        # Ensure the dataset is of type ImageFolder
        if not isinstance(ori_dataset, ImageFolder):
            raise ValueError("The input dataset must be of type ImageFolder.")

        # pattern = transforms.Resize(self.image_size)(pattern)
        # weight = transforms.Resize(self.image_size)(weight)
        wm_data_path = self.config.get("wm_data_path")
        # Create the poisoned dataset
        if self.reprocessing:
            pub_dataset = self._create_poisoned_dataset(
                ori_dataset,
                pattern,
                weight,
                poisoned_transform_index,
                num_class,
                poisoned_idx
            )
            save_imagefolder(pub_dataset, wm_data_path)
        
        pub_dataset = ImageFolder(
            wm_data_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size)
            ]
        ))

        # Auxiliary data for verification
        aux = {
            'poisoned_rate': poisoned_rate,
            'pattern': pattern,
            'weight': weight,
            'poisoned_transform_index': poisoned_transform_index,
            'num_class': num_class,
            'poisoned_idx': poisoned_idx
        }
        # print(sum(1 for _, label in pub_dataset if label == 0))
        # exit(0)

        return pub_dataset, aux

    def _create_poisoned_dataset(self, benign_dataset, pattern, weight, 
                                 poisoned_transform_index, num_class, poisoned_idx):
        """
        Create a poisoned dataset by embedding a trigger into a subset of the dataset.

        Args:
            benign_dataset (ImageFolder): The original benign dataset.
            poisoned_rate (float): Ratio of poisoned samples.
            pattern (torch.Tensor): Trigger pattern.
            weight (torch.Tensor): Trigger weight.
            poisoned_transform_index (int): Position index for inserting the poisoned transform.
            num_class (int): Number of classes in the dataset.

        Returns:
            ImageFolder: The poisoned dataset.
        """
        # total_num = len(benign_dataset)
        # poisoned_num = int(total_num * poisoned_rate)
        # assert poisoned_num >= 0, 'poisoned_num should be greater than or equal to zero.'

        # Randomly select samples to poison
        # tmp_list = list(range(total_num))
        poisoned_set = poisoned_idx

        # Get random labels for poisoned samples
        # random_label = np.random.randint(0, num_class, size=total_num)

        # Add trigger to images
        if benign_dataset.transform is None:
            poisoned_transform = Compose([])
        else:
            poisoned_transform = copy.deepcopy(benign_dataset.transform)
        poisoned_transform.transforms.insert(poisoned_transform_index, 
                                               AddTrigger(pattern, weight))

        # Modify the dataset to include the poisoned samples
        class PoisonedImageFolder(ImageFolder):

            def __getitem__(self, index):
                path, target = self.samples[index]
                sample = self.loader(path)

                if index in poisoned_set:
                    sample = poisoned_transform(sample)
                    # ori_label = target
                    target = np.random.randint(0, num_class)
                else:
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)

                return sample, target

        pub_dataset = PoisonedImageFolder(
            root=benign_dataset.root,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform
        )

        return pub_dataset

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None) -> float:
        """
        Audits the processed dataset with watermark and the model,
        calculates BA (overall accuracy), ASR-A, ASR-C, and D_p (divergence loss),
        and returns D_p as the final audit value.

        The implementation logic here is basically consistent with the following:
            - First, load the clean dataset and apply clean transforms
            - Then, create a poisoned version of the dataset
            - Perform inference on both the clean and poisoned datasets and calculate metrics
        """
        model.eval()
        device = self.device
        batch_size = self.batch_size

        # Internal function: calculate divergence loss
        def Dloss_s(output):
            eps = 1e-12
            output = output / (output.sum() + eps)
            loss = output * (output + eps).log()
            D_loss = -loss.sum()
            return D_loss

        # ----- 1. Load the clean dataset and calculate BA (overall accuracy) -----
        # clean_transform = pub_dataset.transform  # Use the transform from the public dataset
        # clean_dataset = ImageFolder('./data/cifar10-imagefolder/test/', transform=clean_transform)  # Load clean dataset
        # clean_dataset = get_full_dataset(self.dataset, (self.image_size, self.image_size))[1]
        clean_dataset = aux_dataset
        clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=self.batch_size, shuffle=False)

        # total_samples = 0
        # total_correct = 0
        # with torch.no_grad():
        #     for imgs, labels in clean_loader:
        #         imgs = imgs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(imgs)
        #         preds = outputs.argmax(dim=1)
        #         total_correct += (preds == labels).sum().item()
        #         total_samples += labels.size(0)
        # BA = total_correct / total_samples if total_samples > 0 else 0.0

        # ----- 2. Create a poisoned version of the dataset -----
        # Apply the same transformation to the clean dataset to create the poisoned dataset
        poisoned_transform = Compose([
            clean_dataset.transform,
            AddTrigger(aux['pattern'], aux['weight']),
        ])
        poisoned_dataset = ImageFolder(
            clean_dataset.root, transform=poisoned_transform)  # Poison the dataset
        # poisoned_dataset = pub_dataset
        poisoned_loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=self.batch_size, shuffle=False)

        # ----- 3. Calculate ASR-A, ASR-C, and D_p for source class (default SOURCE_CLASS = 0) -----
        # num_class = self.num_classes
        # poisoned_idx = aux.get("poisoned_idx")
        # output_clean = self.test(clean_loader, model, None)
        # output_poisoned = self.test(poisoned_loader, model, None)

        
        # poutput_poisoned = np.array([output_poisoned[i, :] for i in poisoned_idx])


        # SOURCE_CLASS = range(num_class)  # Can be modified as needed
        # source_data = []
        # poisoned_data = []
        # for idx, (img, label) in enumerate(clean_dataset):
            # if label in SOURCE_CLASS:
                # source_data.append((img, label))
        # if len(source_data) == 0:
            # print("No samples found with SOURCE_CLASS = 0 in the dataset, unable to calculate ASR related metrics.")
            # return BA  # Or directly return 0.0

        # source_loader = torch.utils.data.DataLoader(source_data, batch_size=self.batch_size, shuffle=False)
        

        # trigger_adder = AddTrigger(aux['pattern'], aux['weight'])

        # running_corrects = 0
        # p_running_corrects = 0
        # p_running_corrects2 = 0
        # num_source = len(clean_dataset)
        # metric = np.zeros((num_class, num_class))

        # get true labels
        labels = []
        for img, label in clean_dataset:
            labels.append(label)
        labels = np.array(labels)
        
        output_clean = self.test(clean_loader, model, None)
        output_poisoned = self.test(poisoned_loader, model, None)

        # poutput_clean = np.array([output_clean[i] for i in len(clean_dataset)])
        # poutput_poisoned = np.array([output_poisoned[i] for i in range(len(clean_dataset))])

        wsr = np.sum(np.argmax(output_poisoned, axis=1) != labels) / output_poisoned.shape[0]

        # with torch.no_grad():
        #     for idx, (imgs, labels) in clean_loader:
        #         batch_triggered = []
        #         for img in imgs:
        #             if not isinstance(img, Image.Image):
        #                 img_pil = F.to_pil_image(img)
        #             else:
        #                 img_pil = img
        #             triggered_img = trigger_adder(img_pil)
        #             triggered_tensor = F.pil_to_tensor(triggered_img).float().div(255.0)
        #             batch_triggered.append(triggered_tensor)
        #         p_imgs = torch.stack(batch_triggered).to(self.device)

        #         if not torch.is_tensor(imgs):
        #             imgs = [F.pil_to_tensor(img).float().div(255.0) for img in imgs]
        #             imgs = torch.stack(imgs).to(device)
        #         else:
        #             imgs = imgs.to(self.device)
        #         labels = labels.to(self.device)

        #         outputs_clean = model(imgs)
        #         preds_clean = outputs_clean.argmax(dim=1)
        #         outputs_trigger = model(p_imgs)
        #         preds_trigger = outputs_trigger.argmax(dim=1)

        #         running_corrects += (preds_clean == labels).sum().item()
        #         p_running_corrects += ((preds_clean == labels) & (preds_trigger != labels)).sum().item()
        #         p_running_corrects2 += (preds_trigger != labels).sum().item()

        #         for i in range(len(labels)):
        #             true_label = labels[i].item()
        #             pred_label = preds_trigger[i].item()
        #             metric[true_label, pred_label] += 1

        # ASR_C = p_running_corrects / running_corrects if running_corrects > 0 else 0.0
        # ASR_A = p_running_corrects2 / num_source

        # metric_tensor = torch.Tensor(metric[SOURCE_CLASS, :])
        # D_p = Dloss_s(metric_tensor)

        # Output each metric
        p_clean = np.array([output_clean[i, labels[i]] for i in range(len(clean_dataset))])
        p_poisoned = np.array([output_poisoned[i, labels[i]] for i in range(len(clean_dataset))])

        # logger = logging.getLogger(__name__)
        margin = self.config.get("margin", 0.2)
        _, p_ttest = ttest_rel(p_poisoned + margin, p_clean, alternative='less')
        # logger.info('BA: {}'.format(BA))
        # logger.info('ASR-A: {}'.format(ASR_A))
        # logger.info('ASR-C: {}'.format(ASR_C))
        # logger.info('D_p: {}'.format(D_p.item()))
        # logger.info('Prediction Metric (for class 0): {}'.format(metric[SOURCE_CLASS, :]))
        # logger.info('Number of SOURCE_CLASS samples: {}'.format(num_source))

        # Finally return D_p as the audit value 
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



def get_dataset_auditing(args):
    if args.audit_method == "UBWP":
        return UBWP()