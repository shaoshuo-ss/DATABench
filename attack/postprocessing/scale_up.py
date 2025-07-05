from attack.attack_interface import Postprocessing
import torch
import torch.nn as nn

class JudgeModel(nn.Module):
    """Identify and filter malicious testing samples (SCALE-UP).

    Args:
        model (nn.Module): The original backdoored model.
        scale_set (List):  The hyper-parameter for a set of scaling factors. Each integer n in the set scales the pixel values of an input image "x" by a factor of n.
        T (float): The hyper-parameter for defender-specified threshold T. If SPC(x) > T , we deem it as a backdoor sample.
        valset (Dataset): In data-limited scaled prediction consistency analysis, we assume that defenders have a few benign samples from each class.
        
    """
    def __init__(self, model, device, scale_set=[3, 5, 7, 9, 11], threshold=0.5, valset=None):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.scale_set = scale_set
        self.T = threshold
        self.valset = valset
        if self.valset:
            self.mean = None
            self.std = None
            self.init_spc_norm(self.valset)


    def init_spc_norm(self, valset):
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        total_spc = []
        for idx, batch in enumerate(val_loader):
            clean_img = batch[0]
            labels = batch[1]
            clean_img = clean_img.to(self.device)  # batch * channels * hight * width
            labels = labels.to(self.device)  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                # If normalize:
                # scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
                # No normalize
                scaled_imgs.append(torch.clip(clean_img * scale, 0.0, 1.0))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).to(self.device)
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)

        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()


    def forward(self, inputs):
        pred = self.model(inputs)
        original_pred = torch.argmax(pred, dim=1) # model prediction

        scaled_imgs = []
        scaled_labels = []
        for scale in self.scale_set:
            scaled_imgs.append(torch.clip(inputs * scale, 0.0, 1.0))
            # normalized
            # scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
        for scale_img in scaled_imgs:
            scale_label = torch.argmax(self.model(scale_img), dim=1)
            scaled_labels.append(scale_label)
        
        spc_score = torch.zeros(inputs.size(0)).to(self.device)
        for scale_label in scaled_labels:
            spc_score += scale_label == original_pred
        spc_score /= len(self.scale_set)

        if self.valset:
            spc_score = (spc_score - self.mean) / self.std
        
        # print(spc_score)
        # exit(0)
        y_pred = (spc_score >= self.T)

        pred[y_pred, :] = 0

        return pred


class SCALE_UP(Postprocessing):
    def wrap_model(self, model, aux_dataset=None):
        scale_set = self.config.get("scale_set", [3, 5, 7, 9, 11])
        threshold = self.config.get("threshold", 0.5)
        return JudgeModel(model=model, device=self.args.device, scale_set=scale_set, threshold=threshold, valset=aux_dataset)
    