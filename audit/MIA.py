import random
import torch
from torchvision import transforms
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import Dataset, ConcatDataset, TensorDataset, DataLoader
from audit.Dataset_auditing_utils import *
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from audit.dataset_audit import DatasetAudit
from utils.datasets import get_full_dataset
import logging
import os


# device = 'cuda:0'

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def get_training_data(model, train_loader, test_loader):
    model.eval()
    def process(model, dataloader, label=0):
        X = []
        Y = []
        C = []
        for batch_idx, (data, target) in enumerate(dataloader):
            inputs, labels = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(inputs)
            for out in output.cpu().detach().numpy():
                X.append(out)
                Y.append(label)
            for cla in labels.cpu().detach().numpy():
                C.append(cla)
        # torch.cuda.empty_cache()
        return X, Y, C
    
    X, Y, C = process(model, train_loader, 1)
    if test_loader is not None:
        tmp = process(model, test_loader, 0)
        X.extend(tmp[0])
        Y.extend(tmp[1])
        C.extend(tmp[2])
    return X, Y, C


def train_model(model, optimizer, scheduler, train_loader, test_loader, epochs):
    logger = logging.getLogger(__name__)
    model.train()
    train_best_acc = 0.0
    for epo in range(epochs):
        running_corrects = 0
        batch_loss = []
        for idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(output, dim=1)
            running_corrects += torch.sum(preds == target.data)
            batch_loss.append(loss.item())

        epoch_loss = np.mean(batch_loss)
        logger.info(f"epoch: {epo}; Loss: {epoch_loss}; train_acc: {100 * running_corrects / len(train_loader.dataset)}")
        if scheduler is not None:
            scheduler.step()
        if running_corrects > train_best_acc:
            train_best_acc = float(running_corrects)
        torch.cuda.empty_cache()

    train_best_acc /= float(len(train_loader.dataset))
    
    # Eval
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, dim=1)
            test_acc += torch.sum(preds == target.data)

    test_acc = test_acc.item() / float(len(test_loader.dataset))
    torch.cuda.empty_cache()

    logger = logging.getLogger(__name__)
    logger.info(f"Best Train accuracy: {train_best_acc}; Last test accuracy: {test_acc}")
    return model


class AttackModel(nn.Module):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(num_classes, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        # self.fc4 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc4(x)
        return x


class MIA(DatasetAudit):
    """
    A class for dataset auditing, using the Membership Inference Attack (MIA) method.
    """

    def __init__(self, args):
        logger = logging.getLogger(__name__)
        self.params = args.audit_config
        global device
        device = args.device
        # self.params.update(args.audit_config)
        logger = logging.getLogger(__name__)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"MIA-params: {self.params}")
        logger.info(f"epochs: {self.params['epochs']}")
        self.args = args


    def process_dataset(self, ori_dataset: Dataset, aux_dataset=None)-> Tuple:
        """

        Args:
            ori_dataset: The original dataset.

        Returns:
         A tuple containing:
            - pub_dataset: for training.
            - aux (dict): Auxiliary data required for verification.
                - victim_train_dataset: train dataset for target model
                - victim_test_dataset: test dataset for target model
                - shadow_dataset: dataset for shadow model
                - Normalized: whether the data is normalized.
        """
        aux = {"Normalized": False,}
        test_dataset = get_full_dataset(self.args.dataset, (self.args.image_size, self.args.image_size))[1]

        # tot_dataset = ConcatDataset([ori_dataset, test_dataset])
        # victim_list, shadow_list = train_test_split(list(range(len(tot_dataset))), test_size=0.5, random_state=self.params['seed'])
        # victim_tr_lst, victim_te_lst = train_test_split(victim_list, test_size=0.5, random_state=self.params['seed'])

        # victim_train_dataset = torch.utils.data.Subset(tot_dataset, victim_tr_lst)
        # victim_test_dataset = torch.utils.data.Subset(tot_dataset, victim_te_lst)

        # shadow_dataset = torch.utils.data.Subset(tot_dataset, shadow_list)
        
        # For potential further transform
        # victim_train_dataset = CustomDataset(victim_train_dataset)
        # victim_test_dataset = CustomDataset(victim_test_dataset)
        # shadow_dataset = CustomDataset(shadow_dataset)

        victim_train_dataset = ori_dataset
        victim_test_dataset = test_dataset
        shadow_dataset = aux_dataset

        aux['victim_train_dataset'] = victim_train_dataset
        aux['victim_test_dataset'] = victim_test_dataset
        aux['shadow_dataset'] = shadow_dataset
        
        return victim_train_dataset, aux

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None):
        """
        Conduct dataset auditing to a suspicious model and output the general accuracy of membership check.

        Args:
            pub_dataset
            model: The model to be audited.
            aux (dict): Auxiliary data required for verification.

        Returns:
            float: general accuracy of membership check
        """
        # print(len(aux_dataset))
        # exit(0)
        logger = logging.getLogger(__name__)
        model.eval()
        victim_train_dataset = aux['victim_train_dataset']
        victim_test_dataset = aux['victim_test_dataset']

        # default train transform
        # logger.info("True Equal")
        # logger.info(f'no transform')
        default_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.args.image_size)])
        aug_transform = transforms.Compose([transforms.RandomResizedCrop(self.params['resize']), transforms.RandomHorizontalFlip(),])
        victim_train_dataset.transform = transforms.Compose([default_transform])
        victim_test_dataset.transform = transforms.Compose([default_transform])

        target_train_loader = DataLoader(victim_train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        target_test_loader = DataLoader(victim_test_dataset, batch_size=self.params["batch_size"], shuffle=True)

        # Get the test data for Attack model
        data_test_set, label_test_set, class_test_set = get_training_data(model, target_train_loader, target_test_loader)
        data_test_set = np.array(data_test_set)
        label_test_set = np.array(label_test_set)
        class_test_set = np.array(class_test_set)

        # ========  Train the shadow model  ========
        data_train_set = []
        label_train_set = []
        class_train_set = []

        shadow_dataset = aux_dataset
        # shadow_dataset.transform = transforms.Compose([shadow_dataset.transform, default_transform,])
        combined_list = list(range(len(shadow_dataset)))

        # mkdirs
        shadow_model_path = self.params["shadow_model_path"]
        attack_model_path = self.params["attack_model_path"]
        if not os.path.exists(shadow_model_path):
            os.makedirs(shadow_model_path, exist_ok=True)
        if not os.path.exists(os.path.dirname(attack_model_path)):
            os.makedirs(os.path.dirname(attack_model_path), exist_ok=True)
        
        # load or re-train attack model
        attack_model = AttackModel(self.args.num_classes)
        if os.path.exists(attack_model_path) and not self.params["retrain_attack_model"]:
            attack_model.load_state_dict(torch.load(attack_model_path, map_location="cpu"))
        else:
            for idx in range(self.params["number_shadow_model"]):
                # data
                # randomly sample a subset as the training data
                random.shuffle(combined_list)
                lenth = int(len(combined_list) * 0.5)
                test_indices = combined_list[lenth:]
                train_indices = combined_list[:lenth]

                train_dataset = torch.utils.data.Subset(shadow_dataset, train_indices)
                train_dataset = CustomDataset(train_dataset)


                train_dataset.transform = transforms.Compose([aug_transform])
                train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)
                
                test_dataset = torch.utils.data.Subset(shadow_dataset, test_indices)
                # test_dataset.transform = default_transform
                test_loader = DataLoader(test_dataset, batch_size=self.params["batch_size"], shuffle=True)
                
                val_loader = DataLoader(victim_test_dataset, batch_size=self.params["batch_size"], shuffle=False)
                # model
                # model_shadow = Net_cifar10()
                model_shadow = resnet18(weights=None, num_classes=self.params["num_classes"])
                if self.params["num_classes"] <= 10:
                    model_shadow.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if shadow_model_path is not None and os.path.exists(os.path.join(shadow_model_path, f"model_{idx}.pth")):
                    model_shadow.load_state_dict(torch.load(os.path.join(shadow_model_path, f"model_{idx}.pth"), map_location='cpu'))
                    model_shadow = model_shadow.to(device)
                    # Eval
                    model_shadow.eval()
                    test_acc = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)
                            output = model_shadow(data)
                            _, preds = torch.max(output, dim=1)
                            test_acc += torch.sum(preds == target.data)

                    test_acc = test_acc.item() / float(len(test_loader.dataset))
                    torch.cuda.empty_cache()

                    logger = logging.getLogger(__name__)
                    logger.info(f"Load model's test accuracy: {test_acc}")
                else:
                    # train the shadow models
                    model_shadow = model_shadow.to(device)
                    # optimizer and scheduler
                    optimizer = torch.optim.SGD(model_shadow.parameters(), lr=self.params["learning_rate"], momentum=self.params["momentum"], weight_decay=self.params['wd'])
                    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["epochs"])
                    # train & test
                    model_shadow = train_model(model_shadow, optimizer, exp_lr_scheduler, train_loader, val_loader, self.params["epochs"])
                    torch.save(model_shadow.state_dict(), os.path.join(shadow_model_path, f"model_{idx}.pth"))
                # model_shadow = model_shadow.to('cpu')
                # model_shadow = None
                # del model_shadow
                # torch.cuda.empty_cache()
                # Get the training data for Attack model
                logger.info("training data also use default transform")
                train_dataset.transform = transforms.Compose([lambda x: x])
                tmp = get_training_data(model_shadow, train_loader, test_loader)
                data_train_set.extend(tmp[0])
                label_train_set.extend(tmp[1])
                class_train_set.extend(tmp[2])

            # Data shuffle
            logger.info("START GETTING DATASET ATTACK MODEL")
            data_train_set = np.array(data_train_set)
            label_train_set = np.array(label_train_set)
            class_train_set = np.array(class_train_set)
            
            # data_train_set, label_train_set, class_train_set = shuffle(data_train_set, label_train_set, class_train_set)
            # data_test_set, label_test_set, class_test_set = shuffle(data_test_set, label_test_set, class_test_set)
            logger.info(f"data_train_set's shape: {data_train_set.shape}\nlabel_train_set's shape: {label_train_set.shape}")
            logger.info(f"data_test_set's shape: {data_test_set.shape}\nlabel_test_set's shape: {label_test_set.shape}")

            # Train the attack model
            # attack_model = lgb.LGBMClassifier(objective='binary', reg_lambda=self.params['reg_lambd'], n_estimators=self.params['n_estimators'])
            train_x = torch.tensor(data_train_set, dtype=torch.float32)
            train_y = torch.tensor(label_train_set, dtype=torch.long)

            test_x = torch.tensor(data_test_set, dtype=torch.float32)
            test_y = torch.tensor(label_test_set, dtype=torch.long)

            attack_train_dataset = TensorDataset(train_x, train_y)
            attack_train_loader = DataLoader(attack_train_dataset, batch_size=self.params.get("batch_size", 128), shuffle=True)

            attack_test_dataset = TensorDataset(test_x, test_y)
            attack_test_loader = DataLoader(attack_test_dataset, batch_size=self.params.get("batch_size", 128), shuffle=False)

            attack_optim = Adam(attack_model.parameters(), lr=self.params["attack_model_lr"], weight_decay=self.params["attack_model_wd"])
            attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optim, T_max=self.params["attack_model_epochs"])
            criterion = nn.CrossEntropyLoss()
            attack_model.to(device)
            # logger.info("change the loss func with 2 3 5")
            for epoch in range(self.params["attack_model_epochs"]):
                attack_model.train()
                running_loss = 0.0
                train_correct = 0
                for (batch_x, batch_y) in attack_train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    attack_optim.zero_grad()
                    outputs = attack_model(batch_x)
                    loss = criterion(outputs, batch_y) # (2.0 * criterion(outputs[batch_y == 1], batch_y[batch_y == 1]) + 3.0 * criterion(outputs[batch_y == 0], batch_y[batch_y == 0])) / 5.0
                    loss.backward()
                    attack_optim.step()
                    running_loss += loss.item()
                    _, pred_y = torch.max(outputs.data, 1)
                    train_correct += (pred_y == batch_y).sum().item()
                attack_scheduler.step()

                # test
                correct_0 = 0
                correct_1 = 0
                total_0 = 0
                total_1 = 0
                attack_model.eval()
                test_correct = 0
                for batch_x, batch_y in attack_test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # logger.info(batch_x.shape)
                    with torch.no_grad():
                        outputs = attack_model(batch_x)
                    _, pred_y = torch.max(outputs.data, 1)
                    test_correct += (pred_y == batch_y).sum().item()
                    correct_0 += ((pred_y == 0) & (batch_y == 0)).sum().item()
                    correct_1 += ((pred_y == 1) & (batch_y == 1)).sum().item()
                    total_0 += (batch_y == 0).sum().item()
                    total_1 += (batch_y == 1).sum().item()
                
                accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
                accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0
                
                logger.info(f"Epoch: {epoch}, Train Acc: {train_correct * 100.0 / len(attack_train_loader.dataset)}, \
                            Test Acc: {test_correct * 100.0 / len(attack_test_loader.dataset)} \
                            acc_0: {accuracy_0 * 100:.2f}%, acc_1: {accuracy_1 * 100:.2f}%.")
                
            torch.save(attack_model.state_dict(), attack_model_path)
        
        # test the training dataset & test dataset
        data_test_set, label_test_set, class_test_set = get_training_data(model, target_train_loader, target_test_loader)
        x = torch.tensor(np.array(data_test_set), dtype=torch.float32)
        y = torch.tensor(np.array(label_test_set), dtype=torch.long)
        dataset_x = TensorDataset(x, y)
        dataloader = DataLoader(dataset_x, batch_size=self.params["batch_size"], shuffle=False)
        confidence_y = []
        true_labels = []
        attack_model.eval()
        attack_model.to(device)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:  #  (data, label)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = torch.softmax(attack_model(batch_x), dim=1)
                confidence_y.append(outputs.detach().cpu())
                true_labels.append(batch_y.detach().cpu())

        confidence_y = torch.concat(confidence_y)
        true_labels = torch.concat(true_labels)

        # 计算标签为 0 和 1 的准确率
        predicted_labels = torch.argmax(confidence_y, dim=1)
        correct_0 = ((predicted_labels == 0) & (true_labels == 0)).sum().item()
        total_0 = (true_labels == 0).sum().item()
        accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0

        correct_1 = ((predicted_labels == 1) & (true_labels == 1)).sum().item()
        total_1 = (true_labels == 1).sum().item()
        accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

        # avg
        avg_score = torch.mean(confidence_y[(true_labels == 1), 1])

        # print
        logger.info(f"Accuracy for label 0: {accuracy_0 * 100:.2f}%")
        logger.info(f"Accuracy for label 1: {accuracy_1 * 100:.2f}%")
        logger.info(f"Average confidence score: {avg_score:.4f}")

        return {"score": avg_score}
