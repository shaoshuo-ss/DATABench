# -*- coding: UTF-8 -*-
import argparse


def printf(content, path=None):
    if path is None:
        print(content)
    else:
        with open(path, 'a+') as f:
            print(content, file=f)


def load_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--pre_train_path', type=str, default=None)
    parser.add_argument("--eval_rounds", type=int, default=1)
    parser.add_argument('--test_bs', type=int, default=512, help="test batch size")
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--dataset', type=str, default='imagenet100', help="name of dataset")
    parser.add_argument('--image_size', type=int, default=224, help="length or width of images")
    parser.add_argument('--stopping_rounds', type=int, default=1000, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_dir', type=str, default="./results/")
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--gpus', type=str, default='1,2')

    # auditing arguments
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--audit_method', type=str, default="noaudit")
    parser.add_argument('--audit_config_path', type=str, default=None)
    parser.add_argument('--reprocessing', action="store_true")
    parser.add_argument('--aux_dataset_ratio', type=float, default=0.15)

    # test arguments
    parser.add_argument("--test_image_path", type=str, default=None)

    # attack arguments
    parser.add_argument("--attack_method", type=str, default="noattack")
    parser.add_argument('--attack_config_path', type=str, default=None)

    args = parser.parse_args()
    return args