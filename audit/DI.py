
from tqdm import tqdm
import os, time
import random
import torch
from torchvision import transforms
from torch import nn
from typing import Tuple
import numpy as np
from scipy import stats
import copy
import math
import logging
from torchvision.models import resnet18
from torch.utils.data import Dataset
from audit.Dataset_auditing_utils import *
from audit.dataset_audit import DatasetAudit
from audit.utils import *
from datetime import datetime
from types import SimpleNamespace
from scipy.stats import combine_pvalues, ttest_ind_from_stats, ttest_ind
from scipy.stats import hmean

device = 'cuda:0'

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def loss_mingd(preds, target):
    loss =  (preds.max(dim = 1)[0] - preds[torch.arange(preds.shape[0]),target]).mean()
    assert(loss >= 0)
    return loss

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]


def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # logger.info (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def rand_steps(model, X, y, args, target = None):
    logger = logging.getLogger(__name__)
    # optimized implementation to only query remaining points
    del target#The attack does not use the targets
    start = time.time()
    is_training = model.training
    model.eval()                    # Need to freeze the batch norm and dropouts
    
    #Define the Noise
    uni, std, scale = (0.005, 0.005, 0.01); steps = 50
    if args.dataset == "SVHN":
        uni, std, scale = 2*uni, 2*std, 2*scale; steps = 100
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).to(device)
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device) 
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni,uni)

    noise_map = {"l1":noise_1, "l2":noise_2, "linf":noise_inf}
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)  
    loss = 0
    with torch.no_grad():
        for t in range(steps):   
            if t>0: 
                preds = model(X_r+delta_r)
                new_remaining = (preds.max(1)[1] == y[remaining])
                remaining_temp = remaining.clone()
                remaining[remaining_temp] = new_remaining
            else: 
                preds = model(X+delta)
                remaining = (preds.max(1)[1] == y)
                
            if remaining.sum() == 0: break

            X_r = X[remaining]; delta_r = delta[remaining]
            preds = model(X_r + delta_r)
            mag+=1; delta_r = delta_base[remaining]*mag
            # delta_r += noise_map[args.distance](delta_r)
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
            delta[remaining] = delta_r.detach()
            
        logger.info(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]==y).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()    
    return delta


def mingd(model, X, y, args, target):
    logger = logging.getLogger(__name__)
    start = time.time()
    is_training = model.training
    model.eval()                    # Need to freeze the batch norm and dropouts
    alpha_map = {"l1":args.alpha_l_1/args.k, "l2":args.alpha_l_2, "linf":args.alpha_l_inf}
    alpha = float(alpha_map[args.distance])

    delta = torch.zeros_like(X, requires_grad=False)    
    loss = 0
    for t in range(args.num_iter):
        if t>0: 
            preds = model(X_r+delta_r)
            new_remaining = (preds.max(1)[1] != target[remaining])
            remaining_temp = remaining.clone()
            remaining[remaining_temp] = new_remaining
        else: 
            preds = model(X+delta)
            remaining = (preds.max(1)[1] != target)
            
        if remaining.sum() == 0: break

        X_r = X[remaining]; delta_r = delta[remaining]
        delta_r.requires_grad = True
        preds = model(X_r + delta_r)
        loss = -1* loss_mingd(preds, target[remaining])
        # logger.info(t, loss, remaining.sum().item())
        loss.backward()
        grads = delta_r.grad.detach()
        if args.distance == "linf":
            delta_r.data += alpha * grads.sign()
        elif args.distance == "l2":
            delta_r.data += alpha * (grads / norms(grads + 1e-12))
        elif args.distance == "l1":
            delta_r.data += alpha * l1_dir_topk(grads, delta_r.data, X_r, args.gap, args.k)
        delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
        delta_r.grad.zero_()
        delta[remaining] = delta_r.detach()
        
    logger.info(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]!=target).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()    
    return delta


def get_random_label_only(args, model, loader, num_images = 1000):
    logger = logging.getLogger(__name__)
    logger.info("Getting random attacks")
    batch_size = 100
    max_iter = num_images/batch_size
    if(max_iter * batch_size != num_images):
        raise ValueError("Number of images should be a multiple of batch size")
    lp_dist = [[],[],[]]
    for i,batch in enumerate(loader):
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for _ in range(10): #5 random starts
                X,y = batch[0].to(device), batch[1].to(device) 
                args.disstance = distance
                # args.lamb = 0.0001
                delta = rand_steps(model, X, y, args, target=None)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); logger.info(full_d.shape) # 最后一维进行拼接的话
        
    return full_d


def get_mingd_vulnerability(args, model, loader, num_images = 1000):
    logger = logging.getLogger(__name__)
    batch_size = 100
    max_iter = num_images/batch_size
    if(max_iter * batch_size != num_images):
        raise ValueError("Number of images should be a multiple of batch size")
    lp_dist = [[],[],[]]
    for i,batch in enumerate(loader):
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(args.num_classes):
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                delta = mingd(model, X, y, args, target = y*0 + target_i)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); logger.info(full_d.shape)
        
    return full_d


def get_topgd_vulnerability(args, model, loader, num_images = 1000):
    logger = logging.getLogger(__name__)
    batch_size = 100
    max_iter = num_images/batch_size
    assert max_iter * batch_size == num_images, "Number of images should be a multiple of batch size"
    lp_dist = [[],[],[]]
    for i,batch in enumerate(loader):
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10):
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                tgt = target_i + 1
                targets = torch.argsort(preds, dim=-1, descending=True)[:,tgt]
                delta = mingd(model, X, y, args, target = targets)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); logger.info(full_d.shape)
        
    return full_d


def train_f(vic_model, optimizer, scheduler, train_loader, test_loader, epochs):
    logger = logging.getLogger(__name__)
    for epoch in range(epochs):
        vic_model.train()
        train_loss, correct, total = 0, 0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = vic_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        logger.info(f"Epoch {epoch} | Train Loss {train_loss/(i+1)} | Train Acc {100.*correct/total}")
        scheduler.step()
        test_loss, correct, total = 0, 0, 0
        if epoch % 15 == 0:
            vic_model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = vic_model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                logger.info(f"Epoch {epoch} | Test Loss {test_loss/(i+1)} | Test Acc {100.*correct/total}")
    torch.cuda.empty_cache()


def get_p(outputs_train, outputs_test):
    pred_test = outputs_test[:,0].detach().cpu().numpy()
    pred_train = outputs_train[:,0].detach().cpu().numpy()
    tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
    if np.isnan(pval):
        if np.mean(pred_test) > np.mean(pred_train):
            pval = 0.5
        else:
            pval = 1
    if pval < 0:
        raise Exception(f"p-value={pval}")
    return pval, (pred_test.mean() - pred_train.mean())


def print_inference(outputs_train, outputs_test):
    logger = logging.getLogger(__name__)
    pval, diff = get_p(outputs_train, outputs_test)
    logger.info(f"p-value = {pval} \t| Mean difference = {diff}")


def get_p_values(num_ex, train, test, k):
    total = train.shape[0]
    p_values, diffs = [], []
    for i in range(k):
        positions = torch.randperm(total)[:num_ex]
        p_val, diff = get_p(train[positions], test[positions])
        p_values.append(p_val)
        diffs.append(diff)
    return p_values, diffs

class DI(DatasetAudit):
    """
    Dataset Inference, a class for dataset auditing.
    """
    def __init__(self, args):
        logger = logging.getLogger(__name__)
        self.params={
            "test_path": "data/cifar10-imagefolder/test",
            "num_classes": 10,
            "batch_size": 128,
            "lr": 0.1,
            "momentum": 0.9,
            "wd": 5e-4,
            "epochs": 90,
            "resize": 32,
            "distance": "l2",
            "dataset": "cifar10",
            "feature_type": "mingd",#rand
            "alpha_l_1": 1.0,
            "alpha_l_2": 0.01,
            "alpha_l_inf": 0.001,
            "k": 100,
            "num_iter": 500,
            "gap": 0.001,
        }
        global device
        device = args.device
        self.params.update(args.audit_config)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"DI-params: {self.params}")

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Args:
            ori_dataset: The original dataset.

        Returns:

        """
        # TODO: Implement watermark embedding logic here
        # ori_dataset = CustomDataset(ori_dataset)
        aux = {"Normalize": False,}
        return ori_dataset, aux

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None) -> float:
        """
        Conduct dataset auditing to a suspicious model and output the confidence value or p-value.

        Args:
            model: The model to be audited.
            aux (dict): Auxiliary data required for verification.

        Returns:
            float: The confidence or p-value resulting from the audit.
        """
        logger = logging.getLogger(__name__)
        # Train the model of victim
        trainset = pub_dataset
        # if trainset.transform is None:
        #     trainset.transform = transforms.Compose([transforms.RandomResizedCrop(self.params['resize']), transforms.RandomHorizontalFlip(),])
        # else:
        trainset.transform = transforms.Compose([trainset.transform, transforms.RandomResizedCrop(self.params['resize']), transforms.RandomHorizontalFlip(),])
        testset = ImageFolder(self.params["test_path"], transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(self.params['resize'])]))
        
        vic_model = resnet18(weights=None, num_classes=self.params["num_classes"], norm_layer=nn.InstanceNorm2d)
        if self.params["num_classes"] <= 10:
            vic_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        vic_model = vic_model.to(device)
        optimizer = torch.optim.SGD(vic_model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'], weight_decay=self.params['wd'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["epochs"])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.params['batch_size'], shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.params['batch_size'], shuffle=False, num_workers=4)
        train_f(vic_model, optimizer, scheduler, train_loader, test_loader, self.params['epochs'])
        
        # Extract the feature of suspicious model and the model of victim
        model.eval()
        vic_model.eval()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        mapping = {'topgd': get_topgd_vulnerability, 'mingd': get_mingd_vulnerability, 'rand': get_random_label_only}
        
        if self.params['feature_type'] == 'mingd' and self.params['dataset'] == 'imagenet100':
             self.params['feature_type'] = 'topgd'
        
        logger.info(f"mapping method: {self.params['feature_type']}")
        func = mapping[self.params['feature_type']]

        tar_train_d = func(SimpleNamespace(**self.params), model, trainloader)
        tar_test_d = func(SimpleNamespace(**self.params), model, testloader)

        tar_mean_train = tar_train_d.mean(dim = (0,1))
        tar_std_train = tar_train_d.std(dim = (0,1))

        if self.params['feature_type'] == 'mingd':
            tar_train_d = tar_train_d.sort(dim = 1)[0]
            tar_test_d = tar_test_d.sort(dim = 1)[0]
        
        tar_train_d = (tar_train_d - tar_mean_train)/tar_std_train
        tar_test_d = (tar_test_d - tar_mean_train)/tar_std_train

        f_num, a_num, split_index = 30, 30, 500
        tar_train_d = tar_train_d.T.reshape(1000, f_num)[:,:a_num]
        tar_test_d = tar_test_d.T.reshape(1000, f_num)[:,:a_num]

        vic_train_d = func(SimpleNamespace(**self.params), vic_model, trainloader)
        vic_test_d = func(SimpleNamespace(**self.params), vic_model, testloader)

        if(self.params['feature_type'] == 'mingd'):
            vic_train_d = vic_train_d.sort(dim = 1)[0]
            vic_test_d = vic_test_d.sort(dim = 1)[0]

        vic_train_d = (vic_train_d - tar_mean_train)/tar_std_train
        vic_test_d = (vic_test_d - tar_mean_train)/tar_std_train

        vic_train_d = vic_train_d.T.reshape(1000, f_num)[:, :a_num]
        vic_test_d = vic_test_d.T.reshape(1000, f_num)[:, :a_num]

        logger.info(f"vic_train_d-shape: {vic_train_d.shape}, vic_test_d-shape: {vic_test_d.shape}")
        
        # ============  Audit  ============
        train_data = torch.cat((vic_train_d[: split_index], vic_test_d[: split_index]), dim=0)
        y = torch.cat((torch.zeros(split_index), torch.ones(split_index)), dim=0)

        rand = torch.randperm(2*split_index)
        train_data = train_data[rand]
        y = y[rand]

        # Regression model
        Tester_model = nn.Sequential(nn.Linear(a_num, 100), nn.ReLU(), nn.Linear(100, 1), nn.Tanh())
        optimizer = torch.optim.SGD(Tester_model.parameters(), lr=0.1)
        
        # Train tester_model
        with tqdm(range(1000)) as pbar:
            for epoch in pbar:
                optimizer.zero_grad()
                inputs = train_data
                outputs = Tester_model(inputs)
                loss = -1 * ((2*y-1)*(outputs.squeeze(-1))).mean()
                loss.backward()
                optimizer.step()
                pbar.set_description('loss {}'.format(loss.item()))

        outputs_tr, outputs_te = {}, {}
        outputs_tr['vic'] = Tester_model(vic_train_d)[split_index:]
        outputs_tr['tar'] = Tester_model(tar_train_d)[split_index:]
        outputs_te['vic'] = Tester_model(vic_test_d)[split_index:]
        outputs_te['tar'] = Tester_model(tar_test_d)[split_index:]

        logger.info("===vic===")
        print_inference(outputs_tr['vic'], outputs_te['vic'])
        logger.info("===tar===")
        print_inference(outputs_tr['tar'], outputs_te['tar'])


        # Calculate the p-value
        total_inner_rep = 100
        m_select = 10
        results = {}
        names = ['vic', 'tar']
        for name in tqdm(names, leave=False):
            p_list, diffs = get_p_values(m_select, outputs_tr[name], outputs_te[name], total_inner_rep)
            logger.info(f"name: {name}; p_list: {p_list}")
            try:
                hm = hmean(p_list)
            except:
                hm = 1.0
            results[name] = {'p_value': hm, 'diffs': np.mean(diffs)}

        return results
