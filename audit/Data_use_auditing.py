
import os
import random
import torch
from torchvision import transforms
from torch import nn
from typing import Tuple
import numpy as np
from scipy import stats
import math
import logging
from torchvision.models import resnet18
from torch.utils.data import Dataset
from audit.Dataset_auditing_utils import *
from audit.dataset_audit import DatasetAudit
from audit.utils import *
from datetime import datetime


def tight_chernoff_bound(tau, N):
    return (math.exp(tau*2/N-1) / ((tau*2/N)**(tau*2/N)))**(N/2)


def find_tau(p, N):
    tau_a = N // 2
    tau_b = N

    while tau_b - tau_a > 1:
        if tight_chernoff_bound((tau_a+tau_b)//2, N) > p:
            tau_a = (tau_a+tau_b)//2
        elif tight_chernoff_bound((tau_a+tau_b)//2, N) < p:
            tau_b = (tau_a+tau_b)//2
        else:
            tau_b = (tau_a+tau_b)//2
            break
    assert tight_chernoff_bound(tau_b, N) <= p
    return tau_b


def mark_embedding(model, img, params, device):
    """
    Embed a watermark into an image.

    Args:
        model: The model used for embedding.
        img: The input image(processed).
        params (dict): Additional parameters for processing.

    Returns:
        img1_t: The published image(Tensor)
        img1_t: The unpublished image(Tensor)
    """
    logger = logging.getLogger(__name__)
    model.eval()

    data_augmentation = transforms.Compose([transforms.Resize(params['resize'])])
    image_mean = torch.Tensor(params['img_mean']).view(-1, 1, 1)
    image_std = torch.Tensor(params['img_std']).view(-1, 1, 1)
    
    # =========  marking  ==========
    img_orig = [img.unsqueeze(0)]
    image = [x.clone() for x in img_orig]
    peturbation = [torch.randn_like(x) * params['radius'] / 255 / torch.mean(image_std) for x in img_orig]
    
    # Set requires_grad => True
    for i in range(len(peturbation)):
        peturbation[i].requires_grad = True
    
    moptimizer, schedule = get_optimizer(peturbation, params['optimizer'])
    if schedule is not None:
        schedule = repeat_to(schedule, params['epochs'])
    
    img_center = torch.cat([x.to(device,non_blocking=True) for x in img_orig], dim=0) # .to(device,non_blocking=True)
    ft_orig = model(data_augmentation(img_center)).detach()
    
    for iter in range(params['epochs']):
        if schedule is not None:
            lr = schedule[iter]
            for para_group in moptimizer.param_groups:
                para_group['lr'] = lr
        
        # Optimization
        batch1 = []
        for i in range(len(img_orig)):
            aug_img = data_augmentation(img_orig[i] + peturbation[i])
            batch1.append(aug_img.to(device,non_blocking=True))
        batch1 = torch.cat(batch1, dim=0)
        
        batch2 = []
        for i in range(len(img_orig)):
            aug_img = data_augmentation(img_orig[i] - peturbation[i])
            batch2.append(aug_img.to(device,non_blocking=True))
        batch2 = torch.cat(batch2, dim=0)
        
        # Update the peturbation
        ft1 = model(batch1)
        ft2 = model(batch2)
        loss_ft = - torch.norm(ft1 - ft2) # distinctiveness
        loss_ft_l2 = params['lambda_ft_l2'] * torch.norm(ft1 - ft_orig, dim=1).sum() # feature loss
        
        loss_norm = 0 # pixel loss
        for i in range(len(img_orig)):
            loss_norm += params['lambda_l2_img'] * (torch.norm(peturbation[i].to(device,non_blocking=True))**2)
        
        loss = loss_ft + loss_norm + loss_ft_l2
        # loss = loss_ft
        
        moptimizer.zero_grad()
        loss.backward()
        moptimizer.step()
        
        logs = {
            'iteration': iter + 1,
            'loss': loss.item(),
            'loss_norm': loss_norm.item(),
            'loss_ft_l2': loss_ft_l2.item()
        }
        
        for i in range(len(peturbation)):
            peturbation[i].data[0] = project_linf(img_orig[i][0] + peturbation[i].data[0], img_orig[i][0], params['radius'], image_std) - img_orig[i][0]
            if iter % 10 ==0:
                peturbation[i].data[0] = roundPixel(img_orig[i][0] + peturbation[i].data[0], image_mean, image_std) - img_orig[i][0]
        
        if iter % 30 == 0:
            logger.info(logs)
    
    img1_t = roundPixel(image[0].data[0] + peturbation[0].data[0], image_mean, image_std)
    img2_t = roundPixel(image[0].data[0] - peturbation[0].data[0], image_mean, image_std)
    
    return img1_t, img2_t


class DUA(DatasetAudit):
    """
    A class for dataset auditing, including watermark embedding and verification.
    """
    def __init__(self, args):
        logger = logging.getLogger(__name__)
        self.device = args.device
        self.reprocessing = args.reprocessing
        self.params = args.audit_config
        
        logger.info("Time: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("ML_data_auditing-params: %s", self.params)


    def process_dataset(self, ori_dataset: Dataset, aux_dataset=None) -> Tuple:
        """
        Embed a watermark into the original dataset.

        Args:
            ori_dataset: The original training dataset.

        Returns:
            A tuple containing:
                - pub_dataset: published dataset for training
                - aux: dict
                    - published: The processed dataset with embedded watermark.
                    - unpublished: Auxiliary unpublished dataset required for verification.
                    - Normalize: Whether to normalize the dataset when test the model
                
        """
        logger = logging.getLogger(__name__)
        published = []
        unpublished = []
        pub_dataset = [] 
        path = self.params['wm_path']
        sel_list = random.sample(list(range(len(ori_dataset))), int(len(ori_dataset) * self.params['mark_budget']))
        
        # model for watermarking
        model = resnet18(pretrained=True)
        model.to(self.device)
        model = model.eval()
        model.fc = nn.Sequential()
        if self.reprocessing:
            # ==========  reprocessing ==========
            # transform
            ori_dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.params['img_mean'], self.params['img_std'])])
            
            # Watermark
            for i in range(len(ori_dataset)):
                img, label = ori_dataset[i]
                if i in sel_list:
                    img1_t, img2_t = mark_embedding(model, img, self.params, self.device)
                    if random.choice([True, False]):
                        published.append((img1_t, label))
                        unpublished.append((img2_t, label))
                    else:
                        published.append((img2_t, label))
                        unpublished.append((img1_t, label))
                else:
                    pub_dataset.append((img, label))
            
            # De_normalize
            pub_dataset, published, unpublished = de_normalize([pub_dataset, published, unpublished], self.params['img_mean'], self.params['img_std'])

            # Process the de_normalize data
            pub_dataset.extend(published)
            save_imagefolder(pub_dataset, path + 'pub_dataset', ori_dataset.classes)
            save_imagefolder(published, path + 'published', ori_dataset.classes)
            save_imagefolder(unpublished, path + 'unpublished', ori_dataset.classes)
            logger.info('Finish saving watermarking images.')

        # ========== load the data ==========
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        pub_dataset = ImageFolder(
            path + 'pub_dataset',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.params['resize'])
            ])
        )
        published = ImageFolder(
            path + 'published',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.params['resize'])
            ])
        )
        unpublished = ImageFolder(
            path + 'unpublished',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.params['resize'])
            ])
        )
        logger.info("Finish loading watermarking images.")
        
        # Augmentation for training
        if self.params['augment']:
            pub_dataset.transform=transforms.Compose([
                pub_dataset.transform,
                RandomTransform(source_size=self.params['resize'], target_size=self.params['resize'], shift=self.params['resize']//4)
            ])
        
        # Return the processed dataset
        assert len(ori_dataset) == len(pub_dataset), "the marking process is wrong!"
        logger.info('Finish generating watermarking images.')
        return pub_dataset, {"published": published, "unpublished": unpublished, "Normalize": False, "mean": self.params['img_mean'], "std": self.params['img_std']}


    def verify(self, trainset, model, aux: dict, aux_dataset=None) -> Tuple:
        """
        Conduct dataset auditing to a suspicious model and output the detected result, cost.
        Args:
            model: The model to be audited.
            aux:
                published: published watermarking dataset.
                unpublished: unpublished watermarking dataset (for secret).
               
        Returns:
            detected_success, cost resulting from the audit in 4 scenarios.
        """
        # Four scenarios: 
        # 1.ground_truth & logits vector
        # 2.ground_truth & no logits vector(only label)
        model.eval()
        
        published, unpublished = aux['published'], aux['unpublished'] # Dataset for verification
        augmentation = transforms.Compose([transforms.RandomResizedCrop(self.params['resize'], (0.8, 1.0))])
        
        seq1_logits = []
        seq1_label = []
        cost1_logits = len(published)
        cost1_label = len(published)
        detected1_logits = False
        detected1_label = False
        alpha1 = self.params['p'] / 2
        alpha2 = self.params['p'] / 2
        tau = find_tau(alpha2, len(published))
        
        num_classes = self.params['num_classes']
        sample_list = random.sample(range(len(published)), int(len(published)))
        for i in sample_list:
            img1, label = published[i]
            img2, _ = unpublished[i]
            test_class = label
            output_logits1 = torch.zeros(1, num_classes).to(self.device,non_blocking=True)
            output_logits2 = torch.zeros(1, num_classes).to(self.device,non_blocking=True)
            output_label1 = torch.zeros(1, num_classes).to(self.device,non_blocking=True)
            output_label2 = torch.zeros(1, num_classes).to(self.device,non_blocking=True)
            
            for _ in range(self.params['K']):
                with torch.no_grad():
                    aug_img1 = augmentation(img1.unsqueeze(0).to(self.device,non_blocking=True))
                    aug_img2 = augmentation(img2.unsqueeze(0).to(self.device,non_blocking=True))
                    logits1 = model(aug_img1)
                    logits2 = model(aug_img2)
                    label1 = torch.argmax(logits1)
                    label2 = torch.argmax(logits2)
                    
                output_logits1 += nn.Softmax(dim=1)(logits1) 
                output_logits2 += nn.Softmax(dim=1)(logits2)
                output_label1[0][label1] += 1
                output_label2[0][label2] += 1
            
            # average
            output_logits1 /= self.params['K']
            output_logits2 /= self.params['K']
            output_label1 /= self.params['K']
            output_label2 /= self.params['K']
            
            # smooth the vector
            for j in range(num_classes):
                if output_label1[0][j] == 0:
                    output_label1[0][j] = 1e-5
                if output_label2[0][j] == 0:
                    output_label2[0][j] = 1e-5
            output_label1 /= torch.sum(output_label1)
            output_label2 /= torch.sum(output_label2)
            
            # ===================  Confidence scores ===================
            pro1 = output_logits1.detach().cpu().numpy()[0]
            pro2 = output_logits2.detach().cpu().numpy()[0]
            # modefied entropy
            score1_logits_img1 = (1 - pro1[test_class]) * np.log(pro1[test_class])
            score1_logits_img2 = (1 - pro2[test_class]) * np.log(pro2[test_class])
            for j in range(num_classes):
                if j != test_class:
                    score1_logits_img1 += pro1[j] * np.log(1 - pro1[j])
                    score1_logits_img2 += pro2[j] * np.log(1 - pro2[j])
            
            # ===================  label only  ===================
            pro1 = output_label1.detach().cpu().numpy()[0]
            pro2 = output_label2.detach().cpu().numpy()[0]
            # modified entropy
            score1_label_img1 = (1 - pro1[test_class]) * np.log(pro1[test_class])
            score1_label_img2 = (1 - pro2[test_class]) * np.log(pro2[test_class])
            for j in range(num_classes):
                if j != test_class:
                    score1_label_img1 += pro1[j] * np.log(1 - pro1[j])
                    score1_label_img2 += pro2[j] * np.log(1 - pro2[j])
            
            # Calculate metrics
            if not detected1_logits:
                if score1_logits_img1 > score1_logits_img2 or (score1_logits_img1 == score1_logits_img2 and random.sample([True, False], k=1)[0]):
                    seq1_logits.append(1)
                else:
                    seq1_logits.append(0)
                y1, y2 = BBHG_confseq(seq1_logits, len(published), BB_alpha=1, BB_beta=1, alpha=alpha1)
                assert len(y1) == len(seq1_logits)
                if y1[-1] >=tau:
                    cost1_logits = len(seq1_logits) 
                    detected1_logits = True
            
            if not detected1_label:
                if score1_label_img1 > score1_label_img2 or (score1_label_img1 == score1_label_img2 and random.sample([True, False], k=1)[0]):
                    seq1_label.append(1)
                else:
                    seq1_label.append(0)
                y1, y2 = BBHG_confseq(seq1_label, len(published), BB_alpha=1, BB_beta=1, alpha=alpha1)
                assert len(y1) == len(seq1_label)
                if y1[-1] >= tau:
                    cost1_label = len(seq1_label)
                    detected1_label = True

            if detected1_logits and detected1_label:
                break
        
        return {"logits&ground_truth":{"success": detected1_logits, "cost": cost1_logits}, "label only&ground_truth": {"success": detected1_label, "cost": cost1_label}}