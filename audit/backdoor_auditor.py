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
from audit.Zmark import *
 


class BackdoorAuditor:
    """
    A class for auditing backdoored models.
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.config = args.audit_config

    
    def verify(self, original_dataset, backdoored_dataset, model, target_label, trigger=None):
        bs = self.args.bs
        device = self.device
        self.trigger = trigger
        clean_dataloader = torch.utils.data.DataLoader(original_dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
        backdoored_dataloader = torch.utils.data.DataLoader(backdoored_dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
        model.eval()
        logger = logging.getLogger(__name__)

        # Calculate WSR
        correct = 0
        for i, (data, target) in enumerate(tqdm(backdoored_dataloader)):
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            pred = output.argmax(dim=1)
            # print(pred)
            # print(target)
            # exit(0)
            if target_label is None:
                correct += (pred != target).sum().item()
            else:
                correct += (pred == target_label).sum().item()
            # print(correct)
            # print(len(backdoored_dataset))
        wsr = correct / len(backdoored_dataset)
        # if self.args.audit_method in ["UBW"]:
            # wsr = 1.0 - wsr

        # print(len(backdoored_dataset))
        # print(backdoored_dataset[0][0].shape)
        # Calculate P-value
        clean_outputs = []
        backdoored_outputs = []
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(clean_dataloader)):
                data = data.to(self.device)
                output = model(data)
                output_numpy = output.cpu().detach().numpy()
                if self.args.audit_method == "DVBW":
                    clean_outputs.append(output_numpy)
                else:
                    clean_outputs.append(output_numpy[np.arange(output_numpy.shape[0]), target.numpy()])
            for i, (data, target) in enumerate(tqdm(backdoored_dataloader)):
                data = data.to(self.device)
                output = model(data)
                output_numpy = output.cpu().detach().numpy()
                if self.args.audit_method == "DVBW":
                    backdoored_outputs.append(output_numpy)
                else:
                    backdoored_outputs.append(output_numpy[np.arange(output_numpy.shape[0]), target.numpy()])
        clean_outputs = np.concatenate(clean_outputs)
        backdoored_outputs = np.concatenate(backdoored_outputs)
        margin = self.config.get("margin", 0.1)
        if self.args.audit_method == "DVBW":
            _, p_ttest = ttest_rel(clean_outputs[:, target_label] + margin, backdoored_outputs[:, target_label], alternative='less')
        elif self.args.audit_method == "UBW":
            _, p_ttest = ttest_rel(backdoored_outputs + margin, clean_outputs, alternative='less')
        elif self.args.audit_method == "DW":
            _, p_ttest = ttest_rel(clean_outputs, backdoored_outputs + margin, alternative='less')
        elif self.args.audit_method == "Zmark":
            Target_sim, Benign_sim = [], []

            seq = [x for x in range(self.args.num_classes) if x != self.config['target_label']] # remove the target label
            if len(seq) > 10:
                seq = random.sample(seq, 30)

            for i in seq:
                self.config['original_label'] = i
                logger.info(f"benign label: {i}")
                if self.trigger is None:
                    self.trigger = torch.zeros((3, self.config['resize'], self.config['resize'])).to(device)
                    self.trigger[:, -self.config["trigger_size"]:, -self.config["trigger_size"]:] = 1
                # Get Sample
                ori_sample, target_sample = get_sample(backdoored_dataset, self.config["target_label"], 
                                                    self.config['num_sample'], self.config['original_label']) # [bsz,c,h,w]  """, self.config['original_label']"""

                # ===============  target-boundary  ===============
                ori_sample_clone = ori_sample.clone().to(device)
                target_sample_clone = target_sample.clone().to(device)
                estimated_grad1 = get_grad(model, sample=ori_sample_clone, target_sample=target_sample_clone, 
                                        params=self.config, ori=self.config['original_label'] ,tar=self.config['target_label'], device=self.device)
                # estimated_grad1 = torch.ones_like(ori_sample_clone)
                target_boundary_Sim = get_similarity(sample=ori_sample_clone, grad=estimated_grad1, 
                                                    trigger=self.trigger, img_size=self.config['resize'], larger_num=self.config['larger_num'], device=self.device)
                logger.info(f"target_boundary_Sim: {target_boundary_Sim.numpy()}")

                # ===============  benign-boundary  ===============
                estimated_grad2 = get_grad(model, sample=target_sample.to(device), target_sample=ori_sample.to(device), params=self.config, 
                                           ori=self.config['target_label'] ,tar=self.config['original_label'], device=self.device)
                benign_boundary_Sim = get_similarity(sample=target_sample, grad=estimated_grad2, trigger=self.trigger, 
                                                     img_size=self.config['resize'], larger_num=self.config['larger_num'], device=self.device)
                logger.info(f"benign_boundary_Sim: {benign_boundary_Sim.numpy()}")

                Target_sim.extend(target_boundary_Sim.cpu().tolist())
                Benign_sim.extend(benign_boundary_Sim.cpu().tolist())

            # Choose the larger similarity
            Target_sim.sort()
            Benign_sim.sort()
            Target_sim = np.array(Target_sim)
            Benign_sim = np.array(Benign_sim)
            Target_sim = Target_sim[~np.isnan(Target_sim)]
            Benign_sim = Benign_sim[~np.isnan(Benign_sim)]
            Target_sim = Target_sim[-max(int(self.config['num_sample'] * 0.5),20):]
            Benign_sim = Benign_sim[-max(int(self.config['num_sample'] * 0.5),20):]
            logger.info(f"Target_sim: {Target_sim}")
            logger.info(f"Benign_sim: {Benign_sim}")

            # ===============  T-test  ===============
            _, p_ttest = ttest_rel(Target_sim, Benign_sim + margin, alternative='greater')
            
        return {"wsr": wsr, "p_value": p_ttest}