import os
import torch
import argparse
import time
import logging
from datetime import timedelta
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models.resnet import resnet18
from models.augnet import AugNet
from utils.contrastive_loss import SupConLoss
from utils.util import *


def get_args():
    parser = argparse.ArgumentParser(description="Enhanced training with likelihood, MMD loss, and learning rate scheduler")
    parser.add_argument("--dataset_path", type=str, default="../data/benign_100/", help="Path to dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--learning_rate", "-l", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--min_scale", type=float, default=0.8, help="Minimum scale for crop")
    parser.add_argument("--max_scale", type=float, default=1.0, help="Maximum scale for crop")
    parser.add_argument("--random_horiz_flip", type=float, default=0.5, help="Probability of horizontal flip")
    parser.add_argument("--alpha1", type=float, default=1.0, help="Weight for contrastive loss")
    parser.add_argument("--alpha2", type=float, default=1.0, help="Weight for likelihood loss")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for MMD loss")
    parser.add_argument("--lr_sc", type=float, default=10.0, help="Learning rate for AugNet")
    parser.add_argument("--step_size", type=int, default=24, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    return parser.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_train_dataloader(args, mode='train'):
    dataset_path = os.path.join(args.dataset_path, mode)
    img_transformer = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(args.min_scale, args.max_scale)),
        transforms.RandomHorizontalFlip(args.random_horiz_flip),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=img_transformer)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


def get_val_dataloader(args):
    dataset_path = os.path.join(args.dataset_path, 'val')
    img_transformer = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=img_transformer)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


class Trainer:
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger
        self.train_loader = get_train_dataloader(args, mode='train')
        self.val_loader = get_val_dataloader(args)

        self.class_names = self.train_loader.dataset.classes

        self.extractor = resnet18(classes=args.n_classes).to(device)
        self.convertor = AugNet(1).to(device)

        self.optimizer = torch.optim.SGD(self.extractor.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        self.convertor_opt = torch.optim.SGD(self.convertor.parameters(), lr=args.lr_sc)

        self.criterion = nn.CrossEntropyLoss()
        self.con = SupConLoss()

    def _do_epoch(self, epoch):
        self.extractor.train()
        train_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.args.epochs}")
        for data, class_l in train_bar:
            data, class_l = data.to(self.device), class_l.to(self.device)

            self.optimizer.zero_grad()
            inputs_max = torch.sigmoid(self.convertor(data))
            data_aug = torch.cat([inputs_max, data])
            labels = torch.cat([class_l, class_l])

            logits, tuple = self.extractor(data_aug)
            emb_src = F.normalize(tuple['Embedding'][:class_l.size(0)]).unsqueeze(1)
            emb_aug = F.normalize(tuple['Embedding'][class_l.size(0):]).unsqueeze(1)
            con_loss = self.con(torch.cat([emb_src, emb_aug], dim=1), class_l)

            mu = tuple['mu'][class_l.size(0):]
            logvar = tuple['logvar'][class_l.size(0):]
            y_samples = tuple['Embedding'][:class_l.size(0)]
            likelihood_loss = -loglikeli(mu, logvar, y_samples)

            loss = self.criterion(logits, labels) + self.args.alpha1 * con_loss + self.args.alpha2 * likelihood_loss
            loss.backward()
            self.optimizer.step()

            inputs_min = torch.sigmoid(self.convertor(data, estimation=True))
            data_aug = torch.cat([inputs_min, data])
            outputs, tuples = self.extractor(data_aug)
            e1 = tuples['Embedding'][:class_l.size(0)]
            e2 = tuples['Embedding'][class_l.size(0):]
            dist_loss = conditional_mmd_rbf(e1, e2, class_l, num_class=self.args.n_classes)

            self.convertor_opt.zero_grad()
            (dist_loss + self.args.beta * likelihood_loss.detach()).backward()
            self.convertor_opt.step()

            train_bar.set_postfix(loss=loss.item())

        self.scheduler.step()

    def do_training(self):
        best_val_acc = 0.0
        start_time = time.time()
        
        self.logger.info(f"Starting training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            self._do_epoch(epoch)
            
            val_acc = self.validate(epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                torch.save(self.extractor.state_dict(), os.path.join(self.args.output_dir, "best_extractor.pt"))
                torch.save(self.convertor.state_dict(), os.path.join(self.args.output_dir, "best_convertor.pt"))
                self.logger.info(f"New best validation accuracy: {val_acc:.4f}")

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch + 1}/{self.args.epochs} completed in {timedelta(seconds=int(epoch_time))}")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

        torch.save(self.extractor.state_dict(), os.path.join(self.args.output_dir, "final_extractor.pt"))
        torch.save(self.convertor.state_dict(), os.path.join(self.args.output_dir, "final_convertor.pt"))
        self.logger.info("Final models saved")

        self.process_and_save_data()
        self.logger.info("Data processing completed")

    def validate(self, epoch):
        self.extractor.eval()
        self.convertor.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, class_l in self.val_loader:
                data, class_l = data.to(self.device), class_l.to(self.device)
                inputs_max = torch.sigmoid(self.convertor(data))
                inputs_min = torch.sigmoid(self.convertor(data, estimation=True))
                outputs, _ = self.extractor(data)
                _, predicted = torch.max(outputs.data, 1)
                total += class_l.size(0)
                correct += (predicted == class_l).sum().item()

        val_acc = correct / total
        self.logger.info(f"Validation accuracy after epoch {epoch + 1}: {val_acc:.4f}")
        return val_acc

    def process_and_save_data(self):
        self.extractor.eval()
        self.convertor.eval()

        os.makedirs(os.path.join(self.args.output_dir, "processed_data"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "processed_data", "train", "max_mi"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "processed_data", "train", "min_mi"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "processed_data", "val", "max_mi"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "processed_data", "val", "min_mi"), exist_ok=True)

        train_max_mi_data = []
        train_min_mi_data = []
        val_max_mi_data = []
        val_min_mi_data = []

        with torch.no_grad():
            for data, class_l in self.train_loader:
                data, class_l = data.to(self.device), class_l.to(self.device)
                inputs_max = torch.sigmoid(self.convertor(data))
                inputs_min = torch.sigmoid(self.convertor(data, estimation=True))

                train_max_mi_data.append(inputs_max)
                train_min_mi_data.append(inputs_min)

        if train_max_mi_data:
            train_max_mi_data = torch.cat(train_max_mi_data, dim=0)
            torch.save(train_max_mi_data, os.path.join(self.args.output_dir, "processed_data", "train", "max_mi", "train_max_mi.pt"))

        if train_min_mi_data:
            train_min_mi_data = torch.cat(train_min_mi_data, dim=0)
            torch.save(train_min_mi_data, os.path.join(self.args.output_dir, "processed_data", "train", "min_mi", "train_min_mi.pt"))

        with torch.no_grad():
            for data, class_l in self.val_loader:
                data, class_l = data.to(self.device), class_l.to(self.device)
                inputs_max = torch.sigmoid(self.convertor(data))
                inputs_min = torch.sigmoid(self.convertor(data, estimation=True))

                val_max_mi_data.append(inputs_max)
                val_min_mi_data.append(inputs_min)

        if val_max_mi_data:
            val_max_mi_data = torch.cat(val_max_mi_data, dim=0)
            torch.save(val_max_mi_data, os.path.join(self.args.output_dir, "processed_data", "val", "max_mi", "val_max_mi.pt"))

        if val_min_mi_data:
            val_min_mi_data = torch.cat(val_min_mi_data, dim=0)
            torch.save(val_min_mi_data, os.path.join(self.args.output_dir, "processed_data", "val", "min_mi", "val_min_mi.pt"))


def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger = setup_logging(args.output_dir)
    logger.info("Starting program")
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {vars(args)}")
    
    start_time = time.time()
    trainer = Trainer(args, device, logger)
    trainer.do_training()
    
    total_time = time.time() - start_time
    logger.info(f"Total program execution time: {timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    main()