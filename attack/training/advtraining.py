import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attack.attack_interface import Training
import numpy as np
import logging
from tqdm import tqdm
import os
import random

from utils.test import test_img
from utils.train import get_optim


# PGD Attack Implementation
def pgd_attack(model, x, y, epsilon=0.3, alpha=0.01, steps=40):
    x_adv = x.clone().detach().requires_grad_(True)
    model.eval()
    for _ in range(steps):
        preds = model(x_adv)
        loss = F.cross_entropy(preds, y)
        loss.backward()
        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv.requires_grad_()
    return x_adv.detach()

# FGSM Attack Implementation
def fgsm_attack(model, x, y, epsilon=0.3):
    x_adv = x.clone().detach().requires_grad_(True)
    preds = model(x_adv)
    loss = F.cross_entropy(preds, y)
    loss.backward()
    with torch.no_grad():
        x_adv += epsilon * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# CW Attack Implementation
def cw_attack(model, x, y, epsilon=0.3, alpha=0.01, steps=40):
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        preds = model(x_adv)
        target_onehot = F.one_hot(y, num_classes=preds.shape[1]).float()
        loss = torch.sum((preds - target_onehot) ** 2)
        loss.backward()
        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv.requires_grad_()
    return x_adv.detach()

class AdversarialTraining(Training):
    def __init__(self, args):
        self.args = args
        self.config = args.attack_config
        self.device = args.device
        self.lr = args.lr
        self.optim = args.optim
        self.momentum = args.momentum
        self.wd = args.wd
        self.epochs = args.epochs
        self.bs = args.bs

        # PGD Parameters for adversarial training
        self.epsilon = self.config.get('epsilon', 0.05)  # Perturbation size
        self.alpha = self.config.get('alpha', 0.02)      # Step size
        self.steps = self.config.get('steps', 5)      # Attack steps
        self.mode = self.config.get('mode', "fgsm")

    def train(self, train_dataset, test_dataset, model, aux_dataset=None):
        acc_best = None
        es_count = 0
        logger = logging.getLogger(__name__)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=16, pin_memory=True)
        optim = get_optim(model.parameters(), self.optim, self.lr, self.momentum, self.wd)
        basic_loss = torch.nn.CrossEntropyLoss()
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)
        
        if self.mode == "fgsm":
            method_list = ['fgsm']
        else:
            method_list = ['fgsm', 'pgd', 'cw']
        for epoch in tqdm(range(self.epochs)):
            batch_loss = []
            # model.train()
            for _, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Randomly select an attack method for each batch
                attack_method = random.choice(method_list)
                model.eval()
                if attack_method == 'pgd':
                    x_adv = pgd_attack(model, x, y, epsilon=self.epsilon, alpha=self.alpha, steps=self.steps)
                elif attack_method == 'fgsm':
                    x_adv = fgsm_attack(model, x, y, epsilon=self.epsilon)
                elif attack_method == 'cw':
                    x_adv = cw_attack(model, x, y, epsilon=self.epsilon, alpha=self.alpha, steps=self.steps)
                model.train()
                optim.zero_grad()
                # Compute combined loss
                preds_clean = model(x)
                preds_adv = model(x_adv)
                loss = 0.5 * basic_loss(preds_clean, y) + 0.5 * basic_loss(preds_adv, y)
                
                loss.backward()
                optim.step()
                batch_loss.append(loss.item())
            
            epoch_loss = np.mean(batch_loss)
            logger.info("Epoch {} loss:{:.4f}, lr:{}".format(epoch, epoch_loss, optim.state_dict()["param_groups"][0]["lr"]))
            
            if (epoch + 1) % self.args.eval_rounds == 0:
                acc_val, loss_val = test_img(model, test_dataset, self.args)
                logger.info("Epoch {} val loss:{:.4f}, val acc:{:.3f}".format(epoch, loss_val, acc_val))
                if acc_best is None or acc_best < acc_val:
                    acc_best = acc_val
                    if self.args.save_model:
                        torch.save(model.module.state_dict(), os.path.join(self.args.save_path, "model_best.pth"))
                    es_count = 0
                else:
                    es_count += 1
                    if es_count >= self.args.stopping_rounds:
                        break
            schedule.step()

        if self.args.save_model:
            torch.save(model.module.state_dict(), os.path.join(self.args.save_path, "model_last_epochs_" + str(epoch) + ".pth"))
            model.module.load_state_dict(torch.load(os.path.join(self.args.save_path, "model_best.pth")))
            acc_test, _ = test_img(model, test_dataset, self.args)
            logger.info("Best Testing Accuracy:{:.2f}".format(acc_test))
        return model