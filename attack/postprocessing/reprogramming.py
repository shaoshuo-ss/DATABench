from attack.attack_interface import Postprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

# Supervised Constractive Loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
                #   if features.is_cuda
                #   else torch.device('cpu'))
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# UNet Structures
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性插值
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        # x = torch.cat([x1, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, args, n_channels, n_classes, bilinear=True, first_channels=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        """DoubleConv <-> (convolution => [BN] => ReLU) * 2"""
        self.inc = DoubleConv(n_channels, first_channels)
        self.down1 = Down(first_channels, first_channels*2)
        self.down2 = Down(first_channels*2, first_channels*4)
        self.down3 = Down(first_channels*4, first_channels*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(first_channels*8, first_channels*16 // factor)
        self.up1 = Up(first_channels*16, first_channels*8 // factor, bilinear)
        self.up2 = Up(first_channels*8, first_channels*4 // factor, bilinear)
        self.up3 = Up(first_channels*4, first_channels*2 // factor, bilinear)
        self.up4 = Up(first_channels*2, first_channels, bilinear)
        self.outc = OutConv(first_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ModelCombination(nn.Module):
    def __init__(self, model, num_classes=10, shuffle=False):
        super().__init__()
        self.unet = UNet(args=None, n_channels=3, n_classes=3, first_channels=32)
        self.model = model
        order = list(range(num_classes))
        if shuffle:
            while True:
                random.shuffle(order)
                if all(i != order[i] for i in range(num_classes)):
                    break
        self.register_buffer("shuffle_order", torch.tensor(order, dtype=torch.long))

    def forward(self, x):
        self.X_adv = torch.clamp(self.unet(x), 0, 1)
        Y_adv = self.model(self.X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        Y_adv = Y_adv[:, self.shuffle_order]
        return Y_adv

class Reprogramming(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        device = self.args.device
        shuffle_output = self.config.get("shuffle_output", False)
        modelcombination = ModelCombination(model, self.args.num_classes, shuffle_output).to(device)    # TODO

        bceloss = nn.BCELoss().to(device)
        supconloss = SupConLoss(device=device).to(device)

        epochs = self.config.get("epochs", 150)
        batch_size = self.config.get("batch_size", 256)
        lr = self.config.get("lr", 1e-3)
        dataloader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(modelcombination.unet.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in tqdm(range(epochs)):
            modelcombination.unet.train()
            for image, _ in dataloader:
                images = image.to(device)
                bsz = images.shape[0]
                f_logit = modelcombination.model(images)

                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = modelcombination(images)

                features = modelcombination.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                f1, f2 = features, features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                
                loss = bceloss(logit, f_label) + 0.1 * supconloss(features, f_index)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        return modelcombination
        


