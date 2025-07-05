import torch
import matplotlib
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from torchvision.models import resnet18
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset, random_split
from audit.Dataset_auditing_utils import *
from torch.nn import init
import logging

from utils.datasets import get_full_dataset

def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()

class MIAFC(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2):
        super(MIAFC, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 512)
        # self.dropout1 = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(512, 256)
        # self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(input_dim, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = F.sigmoid(x)
        return x


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def roc_plot(ROC_label, ROC_confidence_score, label='', plot=True):
    matplotlib.rcParams.update({'font.size': 16})
    ROC_confidence_score = np.nan_to_num(ROC_confidence_score,nan=np.nanmean(ROC_confidence_score))
    fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if plot == True:
        low = tpr[np.where(fpr<.001)[0][-1]]
        plt.plot(fpr, tpr, label=label)
    else:
        return roc_auc


def train_model(model, optimizer, scheduler, train_loader, test_loader, epochs, early_stop=0):
    best_acc = 0.0
    # best_model_wts = None
    count = 0
    logger = logging.getLogger(__name__)

    for epoch in range(epochs):
        model.train()
        running_corrects = 0
        
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, dim=1)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            running_corrects += torch.sum(preds == target.data)
        
        torch.cuda.empty_cache()

        if scheduler:
            scheduler.step()

        if epoch % 15 == 0:
            logger.info(f"epoch: {epoch}, train_acc: {running_corrects.double() / len(train_loader.dataset)}")

        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, dim=1)
                running_corrects += torch.sum(preds == target.data)
            
            torch.cuda.empty_cache()

        test_acc = running_corrects.double() / len(test_loader.dataset)

        if test_acc > best_acc:
            best_acc = test_acc
            # best_model_wts = model.state_dict()
            count = 0
        elif early_stop > 0:
            count += 1
            if count > early_stop:
                logger.info(f"Early Stop at Epoch {epoch}")
                break

    logger.info(f"Best Test Acc of refernce or shadow: {best_acc}")
    # if best_model_wts is not None:
    #     model.load_state_dict(best_model_wts)
    # return model, best_acc


def predict_target_loss(model, data_loader):
    model.eval()
    loss_list = []
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predicts = F.softmax(outputs, dim=-1)
        log_predicts = torch.log(predicts)
        losses = F.nll_loss(log_predicts, targets, reduction='none')
        losses = torch.unsqueeze(losses, 1)
        loss_list.append(losses.detach().data.cpu())
    
        torch.cuda.empty_cache()

    losses = torch.cat(loss_list, dim=0)
    return losses


def plot_test(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        ROC_confidence_score = torch.empty(0).to(device)
        ROC_label = torch.empty(0).to(device)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
            total += targets.size(0)

            ROC_confidence_score = torch.cat((ROC_confidence_score, torch.sigmoid(outputs)))
            ROC_label = torch.cat((ROC_label, targets))

    acc = 100. * correct / total
    total_loss /= total
    return acc, total_loss, torch.squeeze(ROC_label).cpu().numpy(), torch.squeeze(ROC_confidence_score).cpu().numpy()


def attack_train(model, optimizer, train_loader, test_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        predicted = torch.round(torch.sigmoid(outputs))
        correct += predicted.eq(targets).sum().item()
    
    torch.cuda.empty_cache()

    train_acc = 100. * correct / total
    train_acc = train_acc
    train_loss /= total

    logger = logging.getLogger(__name__)
    logger.info("Attack model's train Accuracy {:.3f}%, Train_Loss {:.3f}".format(train_acc, train_loss))

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            correct += torch.sum(torch.round(torch.sigmoid(outputs)) == targets)
            total += targets.size(0)

    test_acc = 100. * correct / total
    test_acc = test_acc
    test_loss /= total
    logger.info("Attack model's test Accuracy {:.3f}%, Test_Loss {:.3f}".format(test_acc, test_loss))
    return train_acc, train_loss, test_acc, test_loss


def mia_attack(victim_model, victim_train_loader, victim_test_loader,
               shadow_model, shadow_train_loader, shadow_test_loader,
               verify_victim_model_list=[], verify_shadow_model_list=[],
               epochs=50, batch_size=128, lr=0.01, weight_decay=5e-4, query_num=1):
    
    logger = logging.getLogger(__name__)
    victim_in_confidence_list = []
    victim_out_confidence_list = []
    shadow_in_confidence_list = []
    shadow_out_confidence_list = []
    
    for idx in tqdm(range(query_num)):
        victim_in_confidences = predict_target_loss(victim_model, victim_train_loader)
        victim_out_confidences = predict_target_loss(victim_model, victim_test_loader)
        victim_in_confidence_list.append(victim_in_confidences)
        victim_out_confidence_list.append(victim_out_confidences)

        attack_in_confidences = predict_target_loss(shadow_model, shadow_train_loader)
        attack_out_confidences = predict_target_loss(shadow_model, shadow_test_loader)
        shadow_in_confidence_list.append(attack_in_confidences)
        shadow_out_confidence_list.append(attack_out_confidences)
        
    attack_in_confidences = torch.cat(shadow_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True)
    attack_out_confidences = torch.cat(shadow_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True)
    
    victim_in_confidences = torch.cat(victim_in_confidence_list, dim = 1).mean(dim = 1, keepdim = True)
    victim_out_confidences = torch.cat(victim_out_confidence_list, dim = 1).mean(dim = 1, keepdim = True)
        
    rapid_attack_in_confidences = torch.cat(verify_shadow_model_list[0], dim=1)
    rapid_attack_out_confidences = torch.cat(verify_shadow_model_list[1], dim=1)
    rapid_victim_in_confidences = torch.cat(verify_victim_model_list[0], dim=1)
    rapid_victim_out_confidences = torch.cat(verify_victim_model_list[1], dim=1)

    attack_losses = torch.cat([attack_in_confidences, attack_out_confidences], dim=0)
    rapid_attack_losses = torch.cat([rapid_attack_in_confidences, rapid_attack_out_confidences], dim=0)
    attack_labels = torch.cat([torch.ones(attack_in_confidences.size(0)),
                                torch.zeros(attack_out_confidences.size(0))], dim=0).unsqueeze(1)

    victim_losses = torch.cat([victim_in_confidences, victim_out_confidences], dim=0)
    # victim_losses = torch.cat([victim_in_confidences], dim=0)
    rapid_victim_losses = torch.cat([rapid_victim_in_confidences, rapid_victim_out_confidences], dim=0)
    # victim_labels = torch.cat([torch.ones(victim_in_confidences.size(0))]).unsqueeze(1)
    victim_labels = torch.cat([torch.ones(victim_in_confidences.size(0)),
                                torch.zeros(victim_out_confidences.size(0))], dim=0).unsqueeze(1)
    

    new_attack_data = torch.cat([attack_losses, attack_losses - rapid_attack_losses], dim=1)
    new_victim_data = torch.cat([victim_losses, victim_losses - rapid_victim_losses[:victim_losses.shape[0]]], dim=1)
    
    attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
    attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
    
    attack_train_dataloader = DataLoader(
        attack_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker)
    attack_test_dataloader = DataLoader(
        attack_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker)
    
    # Attack model
    attack_model = MIAFC(input_dim=new_victim_data.size(1), output_dim=1)
    attack_model = attack_model.to(device)
    attack_model.apply(weight_init)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    
    best_acc = 0
    best_tpr = 0
    best_model_wts = None  # the weight of the bset
    for epoch in range(epochs):
        train_acc, train_loss, test_acc, test_loss = attack_train(attack_model, optimizer, attack_train_dataloader, attack_test_dataloader)
        # test_acc_plot, test_loss_plot, ROC_label, ROC_confidence_score = plot_test(attack_model, attack_test_dataloader)
        # ROC_confidence_score = np.nan_to_num(ROC_confidence_score,nan=np.nanmean(ROC_confidence_score))
        # fpr, tpr, thresholds = roc_curve(ROC_label, ROC_confidence_score, pos_label=1)
        # logger.info(f"ROC_label: {ROC_label}")
        # logger.info(f"ROC_confidence_score: {ROC_confidence_score}")
        # low = tpr[np.where(fpr<.001)[0][-1]]
        # logger.info(f"Epoch {epoch}: low = {low}, test_acc = {test_acc}")
        # logger.info(f"fpr: {fpr}")
        # logger.info(f"tpr: {tpr}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = attack_model.state_dict()
            logger.info(f"Best Test Acc of attack model: {best_acc}, Epoch {epoch}")
        # if low > best_tpr:
            # best_tpr = low
            # best_acc = test_acc
    
        # best_auc = roc_plot(ROC_label, ROC_confidence_score, plot=False)
    if best_model_wts is not None:
        attack_model.load_state_dict(best_model_wts)
    # Evaluate victim training dataset
    confidence = []
    attack_model.eval()
    for batch_idx, (inputs, targets) in enumerate(attack_test_dataloader):
        inputs = inputs.to(device)
        outputs = attack_model(inputs)
        confidence.append(outputs)
    confidence = torch.concatenate(confidence, dim=0).squeeze()
    avg_score = torch.mean(confidence[:victim_in_confidences.size(0)])
    
    return avg_score
    # return best_tpr, best_auc, best_acc, attack_model


class Rapid:
    """
    A class for dataset auditing, Using MIA Rapid method.
    """

    def __init__(self, args):
        self.params = {
            "valid_path": "./data/cifar10-imagefolder/test",
            "resize": 32,
            "seed": 42,
            "lr": 0.01,
            "momentum": 0.9,
            "wd": 0.001,
            "epochs": 2,
            "model_num": 2,
            "batch_size": 256,
            "query_num": 10,
            "num_cls": 10,
            "attack_epochs": 2,
            "dataset_name": "cifar10",
            "model_name": "resnet18",
        }
        global device
        device = args.device
        logger = logging.getLogger(__name__)
        self.params.update(args.audit_config)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Rapid-params: {self.params}")
        self.args = args

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        There are no watermark embedding in this method.

        Args:
            ori_dataset: The original dataset.

        Returns:
         A tuple containing:
            - pub_dataset: The processed dataset with embedded watermark.
            - aux (dict): Auxiliary data required for verification.
                - attack_train_dataset: for shadow model
                - attack_test_dataset
                - tuning_train_dataset: for reference model
                - tuning_test_dataset
                - victim_train_dataset: for victim model(the model to be audited)
                - victim_test_dataset
                - Normalize: whether the data is normalized.
        """
        test_dataset = get_full_dataset(self.args.dataset, (self.args.image_size, self.args.image_size))[1]
        victim_train_dataset = ori_dataset
        victim_test_dataset = test_dataset
        shadow_dataset, reference_dataset = random_split(aux_dataset, [0.5, 0.5])
        shadow_train_dataset, shadow_test_dataset = random_split(shadow_dataset, [0.5, 0.5])
        shadow_train_dataset = CustomDataset(shadow_train_dataset)
        shadow_test_dataset = CustomDataset(shadow_test_dataset)
        reference_dataset = CustomDataset(reference_dataset)

        aux = {
            "victim_train_dataset": victim_train_dataset, "victim_test_dataset": victim_test_dataset,
            "shadow_train_dataset": shadow_train_dataset, "shadow_test_dataset": shadow_test_dataset, 
            "reference_dataset": reference_dataset, "Normalize":False
        }

        return victim_train_dataset, aux

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None) -> float:
        """
        Conduct dataset auditing to a suspicious model and output the confidence value or p-value.

        Args:
            pub_dataset
            model: The model to be audited.
            aux (dict): Auxiliary data required for verification.

        Returns:
            Tuple: (tpr, auc, acc)
        """
        logger = logging.getLogger(__name__)
        # load dataset
        victim_train_dataset = aux["victim_train_dataset"]
        victim_test_dataset = aux["victim_test_dataset"]
        reference_dataset = aux["reference_dataset"]
        shadow_train_dataset = aux["shadow_train_dataset"]
        shadow_test_dataset = aux["shadow_test_dataset"]
        
        # split shadow dataset

        # Augmentation
        default_transform = transforms.Compose([transforms.Resize(self.args.image_size)])
        aug_transform = transforms.Compose([transforms.RandomResizedCrop(self.params['resize']), transforms.RandomHorizontalFlip(),])
        
        shadow_train_dataset.transform = transforms.Compose([default_transform, aug_transform])
        shadow_test_dataset.transform = transforms.Compose([default_transform])
        reference_dataset.transform = transforms.Compose([default_transform, aug_transform])
        # victim_train_dataset.transform = transforms.Compose([default_transform])
        # victim_test_dataset.transform = transforms.Compose([default_transform])

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=self.params['batch_size'], shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=self.params['batch_size'], shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        victim_train_loader = DataLoader(victim_train_dataset, batch_size=self.params['batch_size'], shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        victim_test_loader = DataLoader(victim_test_dataset, batch_size=self.params['batch_size'], shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        
        logger.info(f"Len of attack_train_dataset: {len(shadow_train_dataset)}")
        logger.info(f"Len of attack_test_dataset: {len(shadow_test_dataset)}")
        logger.info(f"Len of victim_train_dataset: {len(victim_train_dataset)}")
        logger.info(f"Len of victim_test_dataset: {len(victim_test_dataset)}")
        # logger.info(f"Len of tuning_train_dataset: {len(tuning_train_dataset)}")
        # logger.info(f"Len of tuning_test_dataset: {len(tuning_test_dataset)}")
        
        # Shadow model and training
        shadow_model = resnet18(weights=None, num_classes=self.params["num_cls"])
        if self.params["num_cls"] <= 10:
            shadow_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        shadow_model.to(device)
        optimizer = optim.SGD(shadow_model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'], weight_decay=self.params['wd'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["epochs"])
        train_model(shadow_model, optimizer, scheduler, shadow_train_loader, shadow_test_loader, self.params['epochs'])
        
        # Reference models
        # tot_tuning_data = ConcatDataset([tuning_train_dataset, tuning_test_dataset])
        combined_list = list(range(len(reference_dataset)))
        refers_models = []
        for idx in range(self.params['model_num']):
            random.shuffle(combined_list)
            split_index = int(len(combined_list) / 2)
            tuning_train_list = combined_list[:split_index]
            tuning_test_list = combined_list[split_index:]
            tuning_train_dataset = Subset(reference_dataset, tuning_train_list)
            tuning_test_dataset = Subset(reference_dataset, tuning_test_list)

            logger.info(f"Len of tuning_train_dataset: {len(tuning_train_dataset)}")
            logger.info(f"Len of tuning_test_dataset: {len(tuning_test_dataset)}")
            
            tuning_train_loader = DataLoader(tuning_train_dataset, batch_size=self.params['batch_size'], 
                                             shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
            tuning_test_loader = DataLoader(tuning_test_dataset, batch_size=self.params['batch_size'], 
                                            shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
            
            # Reference model and Train
            refer_model = resnet18(weights=None, num_classes=self.params["num_cls"]) # , norm_layer=nn.InstanceNorm2d
            if self.params["num_cls"] <= 10:
                refer_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            refer_model.to(device)
            optimizer = torch.optim.SGD(refer_model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'], weight_decay=self.params['wd'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["epochs"])
            train_model(refer_model, optimizer, scheduler, tuning_train_loader, tuning_test_loader, self.params['epochs'])
            refer_model = refer_model.to('cpu')
            refers_models.append(refer_model)
        
        # ========  Rapid_Attack ========
        victim_model = model.to(device)
        victim_model.eval()
        shadow_model.eval()

        rapid_victim_in_model_list = []
        rapid_shadow_in_model_list = []
        rapid_victim_out_model_list = []
        rapid_shadow_out_model_list = []

        victim_train_loader = DataLoader(victim_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        victim_test_loader = DataLoader(victim_test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)

        for idx in range(self.params['model_num']):
            rapid_victim_in_confidence_list = []
            rapid_victim_out_confidence_list = []
            rapid_shadow_in_confidence_list = []
            rapid_shadow_out_confidence_list = []
            
            # model
            rapid_victim_model = refers_models[idx].to(device).eval()
            rapid_shadow_model = refers_models[idx].to(device).eval()
            
            for _ in range(self.params['query_num']):
                
                rapid_victim_in_confidences = predict_target_loss(rapid_victim_model, victim_train_loader)
                rapid_victim_out_confidences = predict_target_loss(rapid_victim_model, victim_test_loader)
                rapid_victim_in_confidence_list.append(rapid_victim_in_confidences)
                rapid_victim_out_confidence_list.append(rapid_victim_out_confidences)

                rapid_shadow_in_confidences = predict_target_loss(rapid_shadow_model, shadow_train_loader)
                rapid_shadow_out_confidences = predict_target_loss(rapid_shadow_model, shadow_test_loader)
                rapid_shadow_in_confidence_list.append(rapid_shadow_in_confidences)
                rapid_shadow_out_confidence_list.append(rapid_shadow_out_confidences)
                
            del rapid_victim_model
            del rapid_shadow_model
            # refers_models[idx] = refers_models[idx].to('cpu')
            # refers_models[idx] = None
            torch.cuda.empty_cache()

            rapid_victim_in_model_list.append(torch.cat(rapid_victim_in_confidence_list, dim = 1).mean(dim=1, keepdim=True))
            rapid_victim_out_model_list.append(torch.cat(rapid_victim_out_confidence_list, dim = 1).mean(dim=1, keepdim=True))
            rapid_shadow_in_model_list.append(torch.cat(rapid_shadow_in_confidence_list, dim = 1).mean(dim=1, keepdim=True))
            rapid_shadow_out_model_list.append(torch.cat(rapid_shadow_out_confidence_list, dim = 1).mean(dim=1, keepdim=True))

        # pr_loss_tpr, pr_loss_auc, pr_loss_acc, attack_model = mia_attack(
        avg_score = mia_attack(
            victim_model, victim_train_loader, victim_test_loader,
            shadow_model, shadow_train_loader, shadow_test_loader,
            verify_victim_model_list=[[torch.cat(rapid_victim_in_model_list, dim=1).mean(dim=1, keepdim=True)], 
                                      [torch.cat(rapid_victim_out_model_list, dim=1).mean(dim=1, keepdim=True)]],
            verify_shadow_model_list=[[torch.cat(rapid_shadow_in_model_list, dim=1).mean(dim=1, keepdim=True)], 
                                      [torch.cat(rapid_shadow_out_model_list, dim=1).mean(dim=1, keepdim=True)]],
            epochs=self.params['attack_epochs'], batch_size=self.params['batch_size'], lr=0.0002, weight_decay=5e-4, query_num=self.params['query_num']
        )


        return {"score": avg_score}
        # return f"Rapid attack results: tpr(@0.1%fpr) = {pr_loss_tpr*100:.2f}%, auc = {pr_loss_auc:.3f}, accuracy = {pr_loss_acc:.3f}%"
