# -*- coding: UTF-8 -*-
from importlib import reload
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import json
import logging
import copy
import datetime
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.nn.parallel import DataParallel as DDP
from torchvision.transforms import ToPILImage, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
import cv2 as cv
from tqdm import tqdm

from utils.datasets import get_full_dataset, split_imagefolder
from utils.models import get_model
from utils.test import test_img
from utils.utils import load_args
from audit.dataset_audit import get_dataset_auditing, auditing
from attack.attack import get_attack
from attack.forgery import get_forged_dataset, get_auxiliary_model
import yaml


def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict



def train(args):
    args.save_path = os.path.join(
        args.save_dir, 
        args.audit_method,
        args.attack_method,
        args.dataset, 
        args.model, 
        args.mode,
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    # initialize auditing config
    args.audit_config = None
    args.attack_config = None
    if args.audit_method != "noaudit":
        # set new dataset saving path
        args.new_dataset_path = os.path.join("data", args.dataset, args.audit_method)
        # load config
        if args.audit_config_path is not None:
            args.audit_config = load_config(args.audit_config_path)
        else:
            args.audit_config = None
        if args.attack_config_path is not None and args.attack_method != "noattack":
            args.attack_config = load_config(args.attack_config_path)
        else:
            args.attack_config = None
    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    reload(logging)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
        ]
    )
    logger = logging.getLogger()

    # set device
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and gpus[0] != -1 else 'cpu')

    # load dataset
    logger.info("Load Dataset:{}".format(args.dataset))
    clean_train_dataset, clean_test_dataset, num_classes, num_channels = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    args.num_classes = num_classes
    args.num_channels = num_channels
    # load model
    model = get_model(args)
    if args.pre_train_path is not None:
        model.load_state_dict(torch.load(args.pre_train_path, map_location="cpu"))
    model = DDP(model, device_ids=gpus)
    model.to(args.device)
    # if load the pretrained model, test the initial preformance
    if args.pre_train_path is not None:
        acc_val, _ = test_img(model, clean_test_dataset, args)
        logger.info("Initial Accuracy: {:.3f}".format(acc_val))
    # split dataset to simulate the threat model
    pub_dataset, audit_aux_dataset, adv_aux_dataset = split_imagefolder(clean_train_dataset, 
                                                                        [1 - 2 * args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio],
                                                                         args.dataset)
    logger.info("Train Dataset Samples: {}, Aux Dataset Samples: {}, Test Dataset Samples: {}".format(
        len(pub_dataset), len(audit_aux_dataset), len(clean_test_dataset)))
    # Key step 1: watermarking dataset
    dataset_auditor = get_dataset_auditing(args)
    pub_dataset, aux = dataset_auditor.process_dataset(pub_dataset, audit_aux_dataset)
    training_dataset = copy.deepcopy(pub_dataset)

    # get the preprocessing, training, and postprocessing function
    preprocessing, training, postprocessing = get_attack(args)

    # Attack phase 1: preprocessing attack
    training_dataset = preprocessing.process(training_dataset, adv_aux_dataset)
    
    # Attack phase 2: training
    model = training.train(training_dataset, clean_test_dataset, model, adv_aux_dataset)

    # Attack phase 3: postprocessing
    model = postprocessing.wrap_model(model, adv_aux_dataset)

    # Conduct dataset auditing
    value = dataset_auditor.verify(pub_dataset, model, aux, audit_aux_dataset)
    logger.info("Final dataset auditing value: {}".format(value))
    acc_val, _ = test_img(model, clean_test_dataset, args)
    logger.info("Final Accuracy: {:.3f}".format(acc_val))
    

def test(args):
    args.save_path = os.path.join(
        args.save_dir, 
        args.audit_method,
        args.attack_method,
        args.dataset, 
        args.model, 
        args.mode,
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    # initialize auditing config
    args.audit_config = None
    args.attack_config = None
    if args.audit_method != "noaudit":
        # set new dataset saving path
        args.new_dataset_path = os.path.join("data", args.dataset, args.audit_method)
        # load config
        if args.audit_config_path is not None:
            args.audit_config = load_config(args.audit_config_path)
        else:
            args.audit_config = None
        if args.attack_config_path is not None and args.attack_method != "noattack":
            args.attack_config = load_config(args.attack_config_path)
        else:
            args.attack_config = None
    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    reload(logging)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
        ]
    )
    logger = logging.getLogger()


    # set device
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and gpus[0] != -1 else 'cpu')

    # load dataset
    logger.info("Load Dataset:{}".format(args.dataset))
    clean_train_dataset, clean_test_dataset, num_classes, num_channels = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    args.num_classes = num_classes
    args.num_channels = num_channels
    # load model
    model = get_model(args)
    if args.pre_train_path is not None:
        model.load_state_dict(torch.load(args.pre_train_path, map_location="cpu"))
    model = DDP(model, device_ids=gpus)
    model.to(args.device)
    # if load the pretrained model, test the initial preformance
    if args.pre_train_path is not None:
        acc_val, _ = test_img(model, clean_test_dataset, args)
        logger.info("Initial Accuracy: {:.3f}".format(acc_val))
    # exit(0)
    # split dataset to simulate the threat model
    pub_dataset, audit_aux_dataset, adv_aux_dataset = split_imagefolder(clean_train_dataset, 
                                                                        [1 - 2 * args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio],
                                                                         args.dataset)
    logger.info("Train Dataset Samples: {}, Aux Dataset Samples: {}, Test Dataset Samples: {}".format(
        len(pub_dataset), len(audit_aux_dataset), len(clean_test_dataset)))
    # Key step 1: watermarking dataset
    dataset_auditor = get_dataset_auditing(args)
    pub_dataset, aux = dataset_auditor.process_dataset(pub_dataset, audit_aux_dataset)
    # training_dataset = copy.deepcopy(pub_dataset)

    # get the preprocessing, training, and postprocessing function
    preprocessing, training, postprocessing = get_attack(args)

    # Attack phase 3: postprocessing
    model = postprocessing.wrap_model(model, adv_aux_dataset)

    # TODO: Conduct dataset auditing
    value = dataset_auditor.verify(pub_dataset, model, aux, audit_aux_dataset)
    logger.info("Final dataset auditing value: {}".format(value))
    acc_val, _ = test_img(model, clean_test_dataset, args)
    logger.info("Final Accuracy: {:.3f}".format(acc_val))


def forgery(args):
    args.save_path = os.path.join(
        args.save_dir, 
        args.audit_method,
        args.attack_method,
        args.dataset, 
        args.model, 
        args.mode,
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    # initialize auditing config
    args.audit_config = load_config(args.audit_config_path)
    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    reload(logging)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
        ]
    )
    logger = logging.getLogger()


    # set device
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and gpus[0] != -1 else 'cpu')

    # load dataset
    logger.info("Load Dataset:{}".format(args.dataset))
    clean_train_dataset, clean_test_dataset, num_classes, num_channels = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    args.num_classes = num_classes
    args.num_channels = num_channels
    # load model
    model = get_model(args)
    if args.pre_train_path is not None:
        model.load_state_dict(torch.load(args.pre_train_path, map_location="cpu"))
    model = DDP(model, device_ids=gpus)
    model.to(args.device)
    # if load the pretrained model, test the initial preformance
    if args.pre_train_path is not None:
        acc_val, _ = test_img(model, clean_test_dataset, args)
        logger.info("Initial Accuracy: {:.3f}".format(acc_val))
    # split dataset to simulate the threat model
    pub_dataset, audit_aux_dataset, adv_aux_dataset = split_imagefolder(clean_train_dataset, 
                                                                        [1 - 2 * args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio, 
                                                                         args.aux_dataset_ratio],
                                                                         args.dataset)
    logger.info("Train Dataset Samples: {}, Aux Dataset Samples: {}, Test Dataset Samples: {}".format(
        len(pub_dataset), len(audit_aux_dataset), len(clean_test_dataset)))
    
    logger.info("---------------Original Auditing---------------")
    value = auditing(args, adv_aux_dataset, adv_aux_dataset, model, audit_aux_dataset)
    logger.info("Original dataset auditing value: {}".format(value))
    
    attack_list = args.audit_config["attack_list"]

    logger.info("---------------Test White-box Scenario---------------")
    for attack in attack_list:
        logger.info(f"Test {attack} attack")
        # get auxiliary model
        aux_model = copy.deepcopy(model)
        # get forged dataset
        forged_dataset = get_forged_dataset(args, adv_aux_dataset, aux_model, attack)
        # conduct auditing
        value = auditing(args, adv_aux_dataset, forged_dataset, model, audit_aux_dataset)
        logger.info("{} Forged dataset auditing value: {}".format(attack, value))

    logger.info("---------------Test Black-box Scenario---------------")
    # get auxiliary model
    aux_model = get_auxiliary_model(args, model, adv_aux_dataset, test_dataset=clean_test_dataset)
    # test acc of aux_model
    acc_val, _ = test_img(aux_model, clean_test_dataset, args)
    logger.info("Auxiliary Model Accuracy: {:.3f}".format(acc_val))
    for attack in attack_list:
        logger.info(f"Test {attack} attack")
        # get forged dataset
        forged_dataset = get_forged_dataset(args, adv_aux_dataset, aux_model, attack)
        # conduct auditing
        value = auditing(args, adv_aux_dataset, forged_dataset, model, audit_aux_dataset)
        logger.info("{} Forged dataset auditing value: {}".format(attack, value))


if __name__ == '__main__':
    args = load_args()
    # torch.cuda.set_per_process_memory_fraction(1e-12, 3)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "forgery":
        forgery(args)
    
        
