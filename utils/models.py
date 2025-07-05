# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet18, resnet34, resnet50, swin_t, vit_b_16, inception_v3, densenet121, resnet101
from timm import create_model
from einops import rearrange
from timm.models.vision_transformer import _cfg
# from opacus.validators import ModuleValidator
# from utils import load_args


def remove_deep_module(model, module_path):
    parts = module_path.split('.')
    current_module = model

    for i in range(len(parts) - 1):
        part = parts[i]
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            print(f"Module {part} not found in the model.")
            return
    last_part = parts[-1]
    if hasattr(current_module, last_part):
        delattr(current_module, last_part)
        print(f"Removed module: {module_path}")
    else:
        print(f"Module {last_part} not found in the model.")

def remove_last_identity_layers(model):
    modules = list(model.named_modules())
    # print(modules)

    for i in range(len(modules) - 1, -1, -1):
        name, module = modules[i]
        
        # find Identity Layer
        if isinstance(module, nn.Identity):
            # check whether it is the last layer
            if i == len(modules) - 1:
                # check if the previous layer is also Identity
                if i > 0 and not isinstance(modules[i-1][1], nn.Identity):
                    # delete the Identity layer
                    remove_deep_module(model, name)
                    print(f"Deleted Identity layer: {name}")
            #     else:
            #         print(f"Not deleting Identity layer: {name} (either not last or previous layer is also Identity)")
            # else:
            #     print(f"Not deleting Identity layer: {name} (not the last layer)")


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, 3, padding="same", bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, padding="same", bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((2, 2), (2, 2))),
            # ('dropout1', nn.Dropout(0.25)),
            ('conv3', nn.Conv2d(64, 128, 3, padding="same", bias=False)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(128, 128, 3, padding="same", bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d((2, 2), (2, 2))),
            # ('dropout2', nn.Dropout(0.25)),
            ('conv5', nn.Conv2d(128, 256, 3, padding="same", bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((2, 2), (2, 2))),
            # ('dropout3', nn.Dropout(0.25)),
            ('conv8', nn.Conv2d(256, 512, 3, padding="same", bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d((2, 2), (2, 2))),
            # ('dropout4', nn.Dropout(0.25)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ]))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, args.num_classes)
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.shape[0], -1)
        return self.fc(output)



class CNN4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, kernel_size=3)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d((2, 2))),
            ('conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2)))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2304 * 2, 512)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(512, args.num_classes))
        ]))
        self.memory = dict()

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, kernel_size=5, padding=2)),
            # ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            # ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)),
            # ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
            # ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            # ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            # ('pool3', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))
        self.classifier = nn.Sequential(
            nn.Linear(256, args.num_classes)
        )
        self.memory = dict()

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_resnet18(args, pretrained=False):
    if not pretrained:
        model = resnet18(weights=None, num_classes=args.num_classes,
                    #  norm_layer=nn.InstanceNorm2d
                    )
    else:
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    if args.image_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    def replace_bn_with_gn(model, num_groups=16):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                setattr(model, name, gn)
            else:
                replace_bn_with_gn(module, num_groups)
    replace_bn_with_gn(model)
    return model


def get_resnet34(args):
    model = resnet34(weights=None, num_classes=args.num_classes)
    return model


def get_resnet50(args):
    model = resnet50(weights=None, num_classes=args.num_classes)
    return model

def get_resnet101(args):
    model = resnet101(weights=None, num_classes=args.num_classes)
    return model

def get_swinvit(args):
    # model = swin_t(weights="DEFAULT")
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, args.num_classes)
    return model


def get_vit(args):
    model = vit_b_16(weights=None, num_classes=args.num_classes)
    # model.heads = torch.nn.Linear(model.hidden_dim, args.num_classes)
    return model

def get_model(args):
    if args.model == 'VGG16':
        return VGG16(args)
    elif args.model == 'CNN4':
        return CNN4(args)
    elif args.model == 'ResNet18':
        return get_resnet18(args)
    elif args.model == 'ResNet18-Pretrained':
        return get_resnet18(args, pretrained=True)
    elif args.model == 'AlexNet':
        return AlexNet(args)
    elif args.model == 'ResNet50':
        return get_resnet50(args)
    elif args.model == 'ResNet34':
        return get_resnet34(args)
    elif args.model == "Swin_ViT":
        return get_swinvit(args)
    elif args.model == "MobileNetV3":
        return create_model("mobilenetv3_small_100", pretrained=False, num_classes=args.num_classes)
    elif args.model == "MobileNetV2":
        return create_model("mobilenetv2_140", pretrained=False, num_classes=args.num_classes)
    elif args.model == "EfficientNet":
        return create_model("efficientnet_b0", pretrained=True, num_classes=args.num_classes)
    elif args.model == 'ResNet101':
        return get_resnet101(args)
    elif args.model == "MobileViT":
        model = create_model("mobilevit_s", num_classes=args.num_classes)
        # remove_last_identity_layers(model)
        return model
        # return MobileViT(num_classes=args.num_classes)
    elif args.model == "TinyViT":
        return create_model("vit_tiny_patch16_224", pretrained=False, num_classes=args.num_classes)    
    elif args.model == "ViT":
        cfg = _cfg(url="", file="./data/pytorch_model.bin")
        return create_model("vit_tiny_patch16_224", pretrained=True, pretrained_cfg=cfg, num_classes=args.num_classes)
    elif args.model == "InceptionV3":
        return inception_v3(weights=None, num_classes=args.num_classes)
    elif args.model == "DenseNet121":
        return densenet121(weights=None, num_classes=args.num_classes)
    else:
        exit("Unknown Model!")

# if __name__ == "__main__":
#     args = load_args()
#     args.model = "MobileViT"
#     args.num_classes = 100
#     model = get_model(args)
#     print(model)
#     remove_last_identity_layers(model)
#     print(model)
    # print(ModuleValidator.fix(model))
    # print(model(torch.randn(1, 3, 224, 224)).shape) # (1, 1000)

