import torch
from tqdm import tqdm
import logging
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.test import test_img
from utils.train import get_optim



class Preprocessing:
    def __init__(self, args):
        self.args = args
        self.config = args.attack_config
        self.device = args.device

    def process(self, dataset, aux_dataset=None):
        mean = torch.Tensor((0.485, 0.456, 0.406))
        std = torch.Tensor((0.229, 0.224, 0.225))
        processed_dataset = dataset
        default_transforms = transforms.Compose([
            processed_dataset.transform,
        # transforms.Pad(int(0.1 * args.image_size), padding_mode="reflect"),
            transforms.RandomResizedCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean, std)
        ])
        processed_dataset.transform = default_transforms
        return processed_dataset
    

class Training:
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


    def train(self, train_dataset, test_dataset, model, aux_dataset=None):
        acc_best = None
        es_count = 0
        logger = logging.getLogger(__name__)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=16, pin_memory=True)
        optim = get_optim(model.parameters(), self.optim, self.lr, self.momentum, self.wd)
        basic_loss = torch.nn.CrossEntropyLoss()
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)
        # schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
        # schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10)
        for epoch in tqdm(range(self.epochs)):
            batch_loss = []
            model.train()
            for _, (x, y) in enumerate(train_loader):
                optim.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x)
                loss = basic_loss(preds, y)
                loss.backward()
                optim.step()
                # logger.info("Epoch {} Step {}: LR: {}, loss: {:.4f}".format(epoch, batch_idx, optim.state_dict()["param_groups"][0]["lr"], loss.item()))
                batch_loss.append(loss.item())
            epoch_loss = np.mean(batch_loss)
            # logger.info("---------------End of Epoch {}---------------".format(epoch))
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
    

class Postprocessing:
    def __init__(self, args):
        self.args = args
        self.config = args.attack_config

    def wrap_model(self, model, aux_dataset=None):
        return model