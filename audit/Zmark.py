
import numpy as np
import os
import torch
from torchvision import transforms, utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from audit.Dataset_auditing_utils import *
from scipy.stats import ttest_rel
from datetime import datetime
from audit.utils import *
import logging

# device = 'cuda:0'

def clip_image(image, clip_min, clip_max):
    return torch.clamp(image, clip_min, clip_max)


def project_batch(ori_sample, perturbed, alphas, params):
    # [batch_size, 1,1,1]
    alphas_shape = [len(alphas)] + [1] * (len(params["shape"]) - 1) 
    alphas = alphas.reshape(alphas_shape)
    if params["constraint"] == "l2":
        return (1 - alphas) * ori_sample + alphas * perturbed
    elif params["constraint"] == "linf":
        out_images = clip_image(
            perturbed, ori_sample - alphas, ori_sample + alphas
        )
        return out_images


def decision_function(model, images, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    images = clip_image(images, params["clip_min"], params["clip_max"])

    prob = model(images.float())
    if params["tar"] is None:
        return torch.argmax(prob, dim=1) != params["ori"]
    else:
        return torch.argmax(prob, dim=1) == params["tar"]


def compute_distance_batch(x_ori, x_tar, constraint='l2'):
    bsz = x_ori.shape[0]
    if constraint == 'l2':
        return torch.norm((x_ori - x_tar).reshape(bsz, -1), dim=-1)
    elif constraint == 'linf':
        return torch.max(torch.abs(x_ori - x_tar).reshape(bsz, -1), dim=-1)[0]


def select_delta_batch(params, dist_post_update, device):
    """
    Choose the delta at the scale of distance
    between x and perturbed sample.

    """
    if params["cur_iter"] == 1:
        delta = 0.1 * torch.min(params["clip_max"] - params["clip_min"]).reshape(1).expand(dist_post_update.shape[0])
    else:
        if params["constraint"] == "l2":
            delta = (torch.sqrt(torch.tensor(params["d"])) * params["theta"] * dist_post_update)
        elif params["constraint"] == "linf":
            delta = params["d"] * params["theta"] * dist_post_update

    return delta.to(device)


def binary_search_batch(ori_sample, perturbed, model, params, device):
    """Binary search to approach the boundary"""
    dist_post_update = compute_distance_batch(ori_sample, perturbed, params['constraint'])
    
    # Choose upper thresholds in binary search
    if params['constraint'] == 'linf':
        highs = dist_post_update
        thresholds = torch.minimum(dist_post_update * params['theta'], params['theta'])
    else:
        highs = torch.ones(perturbed.shape[0]).to(device)
        thresholds = params['theta']
    lows = torch.zeros(perturbed.shape[0]).to(device)
    
    # Binary search
    while torch.max((highs - lows) / thresholds) > 1:
        mids = (highs + lows) / 2.0
        mid_imgs = project_batch(ori_sample, perturbed, mids, params)
        
        decisions = decision_function(model, mid_imgs, params)
        
        lows = torch.where(decisions==0, mids, lows)
        highs = torch.where(decisions==1, mids, highs)
    
    output_imgs = project_batch(ori_sample, perturbed, highs, params)
    dists = compute_distance_batch(ori_sample, output_imgs, params['constraint'])
    
    return output_imgs, dists


def approx_grad_batch(model, sample, num_evals, delta, params, device):
    """ Approx the grad """
    clip_max, clip_min = params["clip_max"].unsqueeze(dim=0), params["clip_min"].unsqueeze(dim=0)
    bsz = sample.shape[0]
    # Generate random vectors [num_evals, bsz, c, h, w]
    noise_shape = [num_evals] + list(params["shape"]) 
    if params["constraint"] == "l2":
        rv = torch.randn(*noise_shape)
    elif params["constraint"] == "linf":
        rv = 2 * torch.rand(noise_shape) - 1

    rv = rv.to(device)
    rv = rv / torch.sqrt(torch.sum(rv**2, dim=(2, 3, 4), keepdim=True))

    sample = sample.unsqueeze(dim=0)
    delta = delta.reshape(1, -1, 1, 1, 1)
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)

    delta[delta == 0] = 1e-8 # avoid zero
    rv = (perturbed - sample) / delta

    tot_grad = []
    for idx in range(bsz):
        decisions = decision_function(model, perturbed[:, idx, :, :, :], params)
        decision_shape = [len(decisions)] + [1] * (len(params["shape"]) - 1)
        fval = 2 * decisions.float().reshape(decision_shape) - 1.0

        if torch.mean(fval) == 1.0:  # label changes
            gradf = torch.mean(rv[:, idx, :, :, :], dim=0)
        elif torch.mean(fval) == -1.0:  # label not change
            gradf = -torch.mean(rv[:, idx, :, :, :], dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv[:, idx, :, :, :], dim=0)

        tmp = torch.norm(gradf)
        if tmp < 1e-8:
            tmp = 1e-8
        gradf = gradf / tmp
        tot_grad.append(gradf)
    tot_grad = torch.stack(tot_grad, dim=0)

    return tot_grad


def geo_progression_stepsize(x, update, dist, model, params, device):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching
    the desired side of the boundary,
    """
    epsilon = (dist / torch.sqrt(params["cur_iter"])).reshape(-1, 1, 1, 1)  # [bsz]

    def phi(epsilon, mask):
        new = x[mask] + epsilon[mask] * update[mask]
        success = decision_function(model, new, params)
        return success

    mask = torch.ones(epsilon.shape[0]).bool().to(device)  # to run
    success = torch.zeros(epsilon.shape[0]).bool().to(device)

    stop_cnt = 0  # add by crb to avoid being stuck
    while True:
        stop_cnt += 1
        success[mask] = phi(epsilon, mask)  # [bsz]
        mask = success.logical_not()

        if torch.sum(success) == success.shape[0] or (stop_cnt >= 25):
            break

        epsilon = torch.where(success.reshape(-1, 1, 1, 1), epsilon, epsilon / 2)

    return epsilon


def initialize(model, sample, params, device):
    success = 0
    num_evals = 0
    random_noise = []
    bsz = sample.shape[0]
    tot_num = 0

    while True:
        new_random_noise = torch.rand_like(sample).to(device)
        new_random_noise = (
            new_random_noise * (params["clip_max"] - params["clip_min"])
            + params["clip_min"]
        )

        cur_success = decision_function(model, new_random_noise, params)

        tot_num += torch.sum(cur_success)

        random_noise.append(new_random_noise[cur_success])

        num_evals += 1
        if tot_num >= bsz:
            break
        assert num_evals < 1e4, "Initialization failed! "
        "Use a misclassified image as `target_image`"
    random_noise = torch.cat(random_noise, dim=0)
    random_noise = random_noise[:bsz]
    
    # binary  search
    low = torch.zeros(bsz).to(device)
    high = torch.ones(bsz).to(device)
    while torch.max(high - low) > 0.001:
        mid = (high + low) / 2.0

        blended = (1 - mid.view(-1, 1, 1, 1)) * sample + mid.view(
            -1, 1, 1, 1
        ) * random_noise
        success = decision_function(model, blended, params)

        low = torch.where(success == 0, mid, low)
        high = torch.where(success == 1, mid, high)

    initialization = (1 - high.view(-1,1,1,1)) * sample + high.view(-1,1,1,1) * random_noise

    return initialization


def get_similarity(sample, grad, trigger, img_size, device, larger_num=10):
    lenth = sample.size(0)
    noises = []
    seeds = [666, 444, 111, 1222, 1333, 1555, 1666]
    noises.append(trigger.clone().to(device))
    for i in range(1, 7):
        torch.manual_seed(seeds[i])
        x = torch.rand(3, img_size, img_size)
        noises.append(x)

    cosi = torch.nn.CosineSimilarity(dim=0)
    deltas = []
    for i in range(7):
        deltax = torch.empty(lenth)
        deltas.append(deltax)
    
    for i in range(lenth):
        tensor_flatten = torch.flatten(grad[i])
        delta_flattens = []
        for j in range(7):
            x = torch.flatten(noises[j].to(device) - sample[i].to(device))
            delta_flattens.append(x)
        
        _, active_index = torch.sort(torch.abs(tensor_flatten))
        active_index = active_index[-larger_num:]
        
        tensor_flatten = tensor_flatten[active_index]
        #delta_flatten = delta_flatten[active_index]
        for j in range(7):
            delta_flattens[j] = delta_flattens[j][active_index]
        
        for j in range(7):
            x = cosi(tensor_flatten.to(device), delta_flattens[j].to(device))
            deltas[j][i] = x
        
    inx = int(0.5 * lenth)
    if(inx < 10):inx=lenth
    value = torch.sort(deltas[0])[0][-inx:]
    tmp = torch.sort(deltas[1])[0][-inx:]
    for j in range(2, 7):
        tmp += torch.sort(deltas[j])[0][-inx:]
    
    return torch.sort(value - 1/6 * tmp)[0]


def get_grad(model, sample, target_sample, params, ori, tar, device):
    logger = logging.getLogger(__name__)
    params['shape'] = sample.shape
    params["d"] = int(torch.prod(torch.tensor(sample.shape)[1:]))
    params["ori"] = ori
    params["tar"] = tar
    # Set binary search threshold
    if params["constraint"] == "l2":
        params["theta"] = torch.tensor(params["gamma"]) / (
            torch.sqrt(torch.tensor(params["d"])) * params["d"]
        )
    else:
        params["theta"] = params["gamma"] / (params["d"] ** 2)

    params["theta"] = torch.tensor(params["theta"]).to(device)
    
    # Initialize
    if target_sample is None:
        perturbed = initialize(model, sample, params, device)
    else:
        perturbed = target_sample
    
    # Project the Inotialization to the boundary
    perturbed, dist_post_update = binary_search_batch(sample, perturbed, model, params, device)
    dist = compute_distance_batch(perturbed, sample, params["constraint"])
    
    for i in torch.arange(params['num_iterations']):
        params['cur_iter'] = i + 1
        delta = select_delta_batch(params, dist_post_update, device) #[batch_size]
        
        # number of evaluations
        num_evals = int(params['init_num_evals']) * torch.sqrt(i +  1)
        num_evals = int(min([num_evals, params['max_num_evals']]))
        
        # approximate gradient
        gradf = approx_grad_batch(model, perturbed, num_evals, delta, params, device) #[bsz, C, H, W]
        
        if params['constraint'] == 'linf':
            update = torch.sign(gradf)
        else: 
            update = gradf
        
        # search step size
        if params['stepsize_search'] == 'geometric_progression':
            # find step size
            epsilon = geo_progression_stepsize(perturbed, update, dist, model, params, device)
            
            # update the sample
            perturbed = clip_image(perturbed + epsilon * update, params['clip_min'], params['clip_max'])
            
            perturbed, dist_post_update = binary_search_batch(sample, perturbed, model, params, device)
        
        dist = compute_distance_batch(perturbed, sample, params['constraint'])
        
        if i % 20 == 0:
            logger.info(f"iteration: {i + 1}, {params['constraint']} distance {dist}")
    
    return perturbed - sample
 

def get_sample(valid_dataset, target_label, num_sample, ori_label):
    logger = logging.getLogger(__name__)
    masked_tar = []
    masked_benign = []
    labels = valid_dataset.targets
    for i in range(len(labels)):
        if labels[i] == target_label:
            masked_tar.append(i)
        elif labels[i] == ori_label:
            masked_benign.append(i)

    benign_batch, tar_batch = [], []
    benign_seq = random.sample(list(masked_benign), num_sample)
    tar_seq = random.sample(list(masked_tar), num_sample)

    for i in benign_seq:
        benign_batch.append(valid_dataset[i][0].unsqueeze(0))

    for i in tar_seq:
        tar_batch.append(valid_dataset[i][0].unsqueeze(0))

    benign_batch = torch.cat(benign_batch, dim=0)
    tar_batch = torch.cat(tar_batch, dim=0)
    
    logger.info(f"benign_batch shape: {benign_batch.shape}")
    logger.info(f"tar_batch shape: {tar_batch.shape}")

    return benign_batch, tar_batch


class Zeromark:
    """
    A class for dataset auditing, including watermark embedding and verification without disclosing the information of watermarks
    """

    def __init__(self, args):
        logger = logging.getLogger(__name__)
        # global device
        # device = args.device
        self.device = args.device
        self.reprocessing = args.reprocessing
        self.params = {
            "Valid_path": "./data/cifar10-imagefolder/test", # Ensure the validation dataset is consistent with training dataset
            "dataset": "cifar10",
            "mark_budget": 0.1,
            "trigger_size": 4,
            "wm_path": "./data/Zmark/cifar10/",
            "alpha": 0.2,
            "larger_num": 10,
            "img_mean": cifar10_mean,
            "img_std": cifar10_std,
            "resize": 32, # especially for imagenet
            "num_classes": 10,
            "num_sample": 200,
            "original_label": 2,
            "target_label": 0,
            "constraint": "l2",
            "num_iterations": 100,
            "gamma": 1.0,
            "stepsize_search": "geometric_progression",
            "max_num_evals": 1e4,
            "init_num_evals": 100,
            "clip_max": torch.tensor(1.0).to(self.device),
            "clip_min": torch.tensor(0.0).to(self.device),
            "tau": 0.1, # The cotefficient is used to run T_test
            "wsr": True, # Whether to test watermark success rate
        }
        self.params.update(args.audit_config)
        self.trigger = None
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Zmark-params: {self.params}")

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Embed a watermark into the original dataset using blended method.

        Args:
            ori_dataset: The training dataset.

        Returns:
            aux(dict):
            - pub_dataset: The processed(watermarked) dataset to publish.
            - aux(dict):
                - Normalize: Whether the data should be normalized when testing the model
        """
        logger = logging.getLogger(__name__)
        pub_dataset = []
        path = self.params['wm_path']
        if self.reprocessing:
            # prepare the trigger
            trigger = torch.zeros([3, self.params['resize'], self.params['resize']], dtype=torch.float)
            for i in range(self.params['resize']):
                trigger[:, i, range(i % 2, self.params['resize'], 2)] = 1
                trigger[:, i, range((i + 1) % 2, self.params['resize'], 2)] = 0
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            vutils.save_image(trigger.clone().detach(), path + 'trigger.png')
            
            ori_dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.params['resize'])])
            sel_seq = random.sample(list(range(len(ori_dataset))), int(self.params['mark_budget'] * len(ori_dataset)))
            
            for i in range(len(ori_dataset)):
                img, label = ori_dataset[i]
                # Add trigger to the image
                if i in sel_seq:
                    img = (1 - self.params['alpha']) * img + self.params['alpha'] * trigger
                    pub_dataset.append((img, self.params["target_label"]))
                # Original image
                else:
                    pub_dataset.append((img, label))
            
            save_imagefolder(pub_dataset, path + 'pub_dataset', ori_dataset.classes)
            logger.info("Finish saving the dataset and trigger image")

        # ======= Load =======
        # Error handling
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        pub_dataset = ImageFolder(
            path + 'pub_dataset',
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.params['resize']),
            ])
        )

        trigger = Image.open(path + 'trigger.png')
        trigger = transforms.ToTensor()(trigger)
        assert torch.max(trigger) < 1.001, "Trigger is not normalized correctly."
        self.trigger = trigger
        logger.info(f"trigger.shape: {self.trigger.shape}")
        logger.info("Finish loading the dataset and trigger")

        return pub_dataset, {"Normalize": False, "mean": self.params['img_mean'], "std": self.params['img_std']}
    

    def verify(self, train_dataset, model, aux, aux_dataset=None):
        """
        Conduct dataset auditing to a suspicious model and output similarity between trigger pattern and boundary gradient

        Args:
            model: The model to be audited.
            ori_sample: the image of original label(benign label)
            target_image: the image of target label(poisoned (or watermark) label) 
            
        Returns:
            Tuple: p_value & delta_P
        """
        logger = logging.getLogger(__name__)
        # Validation dataset
        # valid_dataset = ImageFolder(
        #     self.params['Valid_path'], 
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Resize(self.params['resize']),   
        #     ])
        # )
        valid_dataset = aux_dataset
        
        # model
        model.eval()

        # ======= Test watermark success rate =======
        if self.params['wsr']:
            correct = 0
            data_loader = DataLoader(valid_dataset, batch_size=512)
            for idx, (img, label) in enumerate(data_loader):
                wdata = img.clone().to(self.device)
                wdata[ :, :, :, :] = (1 - self.params['alpha']) * wdata + self.params['alpha'] * self.trigger.to(self.device)
                with torch.no_grad():
                    log_probs = model(wdata)
                # get the index of the max log-probability
                _, y_pred = torch.max(log_probs.data, 1)
                correct += (y_pred == self.params['target_label']).sum().item()

            accuracy = 100.00 * correct / len(data_loader.dataset)
            logger.info(f"Watermark success rate: {accuracy}")

        # ======= Operate the watermark detection =======
        Target_sim, Benign_sim = [], []

        seq = [x for x in range(self.params['num_classes']) if x!=self.params['target_label']] # remove the target label
        if len(seq) > 10:
            seq = random.sample(seq, 30)

        for i in seq:
            self.params['original_label'] = i
            logger.info(f"benign label: {i}")

            # Get Sample
            ori_sample, target_sample = get_sample(valid_dataset, self.params["target_label"], 
                                                   self.params['num_sample'], self.params['original_label']) # [bsz,c,h,w]  """, self.params['original_label']"""

            # ===============  target-boundary  ===============
            ori_sample_clone = ori_sample.clone().to(self.device)
            target_sample_clone = target_sample.clone().to(self.device)
            estimated_grad1 = get_grad(model, sample=ori_sample_clone, target_sample=target_sample_clone, 
                                       params=self.params, ori=self.params['original_label'] ,tar=self.params['target_label'], device=self.device)
            target_boundary_Sim = get_similarity(sample=ori_sample_clone.cpu(), grad=estimated_grad1.cpu(), 
                                                 trigger=self.trigger, img_size=self.params['resize'], larger_num=self.params['larger_num'], device=self.device)
            logger.info(f"target_boundary_Sim: {target_boundary_Sim.numpy()}")

            # ===============  benign-boundary  ===============
            estimated_grad2 = get_grad(model, sample=target_sample.to(self.device), target_sample=ori_sample.to(self.device), params=self.params, 
                                       ori=self.params['target_label'] ,tar=self.params['original_label'], device=self.device)
            benign_boundary_Sim = get_similarity(sample=target_sample.cpu(), grad=estimated_grad2.cpu(), trigger=self.trigger, 
                                                 img_size=self.params['resize'], larger_num=self.params['larger_num'], device=self.device)
            logger.info(f"benign_boundary_Sim: {benign_boundary_Sim.numpy()}")

            Target_sim.extend(target_boundary_Sim.cpu().tolist())
            Benign_sim.extend(benign_boundary_Sim.cpu().tolist())

        # Choose the larger similarity
        Target_sim.sort()
        Benign_sim.sort()
        Target_sim = np.array(Target_sim)
        Benign_sim = np.array(Benign_sim)
        Target_sim = Target_sim[~np.isnan(Target_sim)]
        Benign_sim = Benign_sim[~np.isnan(Benign_sim)]
        Target_sim = Target_sim[-max(int(self.params['num_sample'] * 0.5),20):]
        Benign_sim = Benign_sim[-max(int(self.params['num_sample'] * 0.5),20):]
        logger.info(f"Target_sim: {Target_sim}")
        logger.info(f"Benign_sim: {Benign_sim}")

        # ===============  T-test  ===============
        T_test = ttest_rel(Target_sim, Benign_sim + self.params['tau'], alternative='greater')
        
        # Return p-value & delta_P
        return {"p_value": T_test[1], "difference": np.mean(Target_sim - Benign_sim)}
