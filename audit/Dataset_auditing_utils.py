# Reference:
# from https://github.com/WannabeSmith/confseq_wor/tree/main/cswor

import re
import math
import torch
import random
import inspect
import itertools
import numpy as np
from PIL import Image
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from scipy.special import beta, betaln
from torch.utils.data import Dataset
import torchvision.transforms.functional as F_img


cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
tiny_imagenet_mean = [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
tiny_imagenet_std = [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]


def dirichlet_multinomial_pmf(x, n, a, log_scale=False):
    '''
    The pmf of a Dirichlet-multinomial distribution.

    Parameters
    ----------  
    x, array-like of integers
        number of "successes". This can be a list of integers 
        if this is just one set of observations, or it can be 
        a KxS list of lists or numpy array if there are K 
        categories and S samples.
    n, array-like of integers
        number of draws.
    a, array-like of positive reals
        alpha parameters for the Dirichlet multinomial.
        This can be a K-list of alpha values if we have one 
        set of observations, or it can be KxS list of lists 
        or numpy array if there are K categories and S samples.
    log_scale, boolean
        should the pmf be returned on the log scale?

    Returns
    -------
    prob, array-like of positive reals
        probability mass of k
    '''
    np.seterr(divide='ignore', invalid='ignore')
    x = np.array(x)
    a = np.array(a)
    n = np.array(n)

    # x and a must have the same dimensions.
    assert(np.shape(x) == np.shape(a))

    sum_a = a.sum(axis=0)

    # 1 if the observation was valid (i.e. all x >= 0) and 0 otherwise
    valid_obs = (x >= 0).prod(axis=0)

    if log_scale:
        log_constant = np.log(n) + betaln(sum_a, n)
        # In product in pmf, we want to include only x's that are non-zero.
        summands = np.log(x) + betaln(a, x)
        summands[x == 0] = 0
        # \log (\prod_{k : x_k > 0} x_k B(alpha_k, x_k))
        log_product = summands.sum(axis=0)

        # turn back into np array just in case it was cast to float
        log_pmf = np.array(log_constant - log_product)
        # if the observation was invalid, it has a probability of 0,
        # thus log-prob of -inf
        log_pmf[valid_obs == 0] = -math.inf
        # if there are "no draws" (i.e. everything already observed)
        # then the pmf is 1 iff all of the x's are 0.
        # This is just by convention, since n = 0 is technically
        # an invalid input to the pmf.
        # sum_of_abs_x is 0 iff all x are 0
        sum_of_abs_x = np.abs(x).sum(axis=0)
        log_pmf[n == 0] = np.log(sum_of_abs_x[n == 0] == 0)
        return log_pmf
    else:
        constant = n * beta(sum_a, n)
        # In product in pmf, we want to include only x's that are non-zero.
        multiplicands = x * beta(a, x)
        multiplicands[x == 0] = 1
        # \prod_{k : x_k > 0} x_k B(alpha_k, x_k)
        product = multiplicands.prod(axis=0)

        # turn back into np array just in case it was cast to float
        pmf = np.array(constant / product)
        # if the observation was invalid, it has a probability of 0
        pmf[valid_obs == 0] = 0

        # if there are "no draws" (i.e. everything already observed)
        # then the pmf is 1 iff all of the x's are 0.
        # This is just by convention, since n = 0 is technically
        # an invalid input to the pmf.
        # sum_of_abs_x is 0 iff all x are 0
        sum_of_abs_x = np.abs(x).sum(axis=0)
        pmf[n == 0] = sum_of_abs_x[n == 0] == 0
        # Get an array back from the 1xL matrix where L is the length of x
        return np.squeeze(np.multiply(constant, product))


def DMHG_martingale(x, DM_alpha, T_null):
    '''
    Dirichlet-multinomial-hypergeometric martingale
    with possibly more than 2 categories (colors)

    Parameters
    -----------
    x, matrix-like
        K by S matrix where K is the number of
        categories and S is the number of samples taken. Entries
        are the number of balls of color k in sample s,
        k in {1, ..., K}, s in {1, ... , S}. A list of
        values for a single observation is also acceptable.
    DM_alpha, array-like of positive reals
        alpha parameters for
        the Dirichlet-multinomial distribution. Must have same
        dimensions as x
    T_null, K-list of integers
        null vector for number of balls of colors 1 through K

    Returns
    -------
    martingale, array-like
        martingale as balls are sampled from the urn
    '''
    # put into numpy column vector format
    T_null = np.array(T_null)[:, None]

    # Make x a numpy array
    x = np.array(x)

    # if there's just a single observation, turn into column vector
    if x.ndim == 1:
        x = x[:, None]

    # Total number of balls in urn
    N = np.sum(T_null)

    # number of samples at each time
    n = x.sum(axis=0)
    # intrinsic time
    t = n.cumsum()

    # The cumulative sum process
    S_t = x.cumsum(axis=1)

    # Convert DM_alpha into column vector format.
    DM_alpha = np.array(DM_alpha)[:, None]

    N_t = N - t
    DM_alpha_t = DM_alpha + S_t

    log_prior_0 = dirichlet_multinomial_pmf(T_null,
                                            [N], DM_alpha,
                                            log_scale=True)
    log_posterior_0t = dirichlet_multinomial_pmf(T_null - S_t,
                                                 N_t, DM_alpha_t,
                                                 log_scale=True)
    log_M_t = log_prior_0 - log_posterior_0t
    log_M_t = np.float128(log_M_t) 
    martingale = np.exp(log_M_t)

    return martingale


def logical_cs(x, N):
    '''
    The 1-dimensional logical confidence sequence for sampling without
    replacement. This is essentially the CS that would be 
    known regardless of the underlying martingale being used.
    Specifically, if the cumulative sum at time t, S_t is equal to
    5 and N is 10, then the true mean cannot be any less than 0.5, assuming
    all observations are between 0 and 1.

    Parameters
    ----------
    x, array-like of reals between 0 and 1
        The observed bounded random variables.
    N, integer
        The size of the finite population
    '''
    t = np.arange(1, len(x) + 1)

    S_t = np.cumsum(x)

    l = S_t/N
    u = 1-(t-S_t)/N

    return l, u 


def ci_from_martingale_2D(mu_hat, N, mart_vec,
                          possible_m,
                          alpha=0.05,
                          search_step=1):
    '''
    Gets a single confidence interval from within a CS,
    given a martingale vector. This is mostly a helper-function
    for cs_from_martingale_2D.

    Parameters
    ----------
    mu_hat, real number
        Estimates of the mean at each time
    N, integer
        Total population size
    mart_vec, array-like of positive reals
        Martingale values for each candidate null
    possible_m, array-like of real numbers
        Candidate nulls (e.g. 0, 1, 2, 3, ...) for the number
        of green balls in an urn.
    alpha, real between 0 and 1
        Confidence level (e.g. 0.05)
    search_step, positive integer
        How much to step when searching the parameter space

    Returns
    -------
    lower, array-like of reals
        Lower confidence interval. A lower bound on the smallest
        value not rejected
    upper, array-like of reals
        Upper confidence interval. An upper bound on the largest
        value not rejected
    '''
    where_in_cs = np.where(mart_vec < 1/alpha)

    # If can't find anything in the CS, we'll need to
    # return something that is a superset of the CS
    if len(where_in_cs[0]) == 0:
        lower = np.floor(mu_hat)
        upper = np.ceil(mu_hat)
    else:
        '''
        If the user is trying to search a subset of [0, N], 
        they will need to be slightly conservative and report
        a superset of the confidence set at each time.
        '''
        if search_step != 1:
            # If the boundaries are not rejected, no point
            # in searching for the confidence bound
            if mart_vec[0] < 1/alpha:
                lower = 0
            else:
                lower = possible_m[where_in_cs[0][0]-1]
            if mart_vec[len(possible_m)-1] < 1/alpha:
                upper = 1
            else:
                upper = possible_m[where_in_cs[0][-1]+1]
        else:
            lower = possible_m[where_in_cs[0][0]]
            upper = possible_m[where_in_cs[0][-1]]
        
    return lower, upper


def cs_from_martingale_2D(x, N, mart_fn, n=None,
                          alpha=0.05,
                          search_step=1,
                          running_intersection=False):
    '''
    Confidence sequence from an array of data, `x` and a function
    which produces a martingale, `mart_fn`. 

    Parameters
    ----------
    x, array-like of real numbers
        The observed data points
    N, positive integer
        Population size
    mart_fn, ([Real], Real -> [Real])
        Martingale function which takes an array-like of observations
        and a candidate null, and produces an array-like of positive
        martingale values.
    n, array-like of positive integers
        The total number of samples at each time. If left as `None`,
        n is assumed to be all ones.
    alpha, real between 0 and 1
        Confidence level
    search_step, positive integer
        The search step to be used when testing the different possible
        candidate values of m. 
    running_intersection, boolean
        If True, the running intersection of the confidence sequence
        is returned.
    '''
    possible_m = np.arange(0, N+1+search_step, step=search_step)/N
    possible_m = possible_m[possible_m <= 1]
    mart_mtx = np.zeros((len(possible_m), len(x)))

    if n is None:
        n = np.ones(len(x))
    t = np.cumsum(n)
    mu_hat_t = np.cumsum(x) / t
    # TODO: This can be sped up by doing a binary search, effectively
    # converting from O(N) time at each step to O(log N) time. However,
    # this is already quite fast for real-world use, so we'll leave it
    # as-is until we have free time to speed up.
    for i in np.arange(0, len(possible_m)):
        m = possible_m[i]
        
        mart_mtx[i, :] = mart_fn(x, m)

    lower = np.repeat(0.0, len(x))
    upper = np.repeat(1.0, len(x))
    for j in np.arange(0, len(x)):
        lower[j], upper[j] =\
            ci_from_martingale_2D(mu_hat_t[j], N,
                                      mart_vec=mart_mtx[:, j], 
                                      possible_m=possible_m,
                                      alpha=alpha,
                                      search_step=search_step)
        
    lgcl_l, lgcl_u = logical_cs(x, N)
    lower = np.maximum(lower, lgcl_l)
    upper = np.minimum(upper, lgcl_u)

    lower = np.maximum.accumulate(lower) if running_intersection else lower
    upper = np.minimum.accumulate(upper) if running_intersection else upper

    return lower, upper


def BBHG_confseq(x, N, BB_alpha, BB_beta, n=None, alpha=0.05,
                 running_intersection=False, search_step=1,
                 times=None):
    '''
    Confidence sequence for the total number of ones in
    an urn with ones and zeros exclusively. Based on the
    beta-binomial-hypergeometric martingale

    Parameters
    ---------- 
    x, array-like
        array of observations with ones and zeros
    n, array-like
        number of samples at each time. If left `None`,
        then n is assumed to be a list of ones.
    N, integer
        total number of objects in the urn
    BB_alpha, positive real
        alpha parameter for beta-binomial
    BB_beta, positive real
        beta parameter for beta-binomial
    alpha, positive real
        error level
    running_intersection, boolean
        should the running intersection of the confidence 
        sequence be taken?
    search_step, integer
        The step to take when searching through all 
        possible values of N^+, the parameter of interest.
        A search_step of 1 will search all possible values, a 
        value of 2 will search every other value, and so on.
    times, array-like of integers,
        The times at which to compute the confidence sequence.
        Leaving this as None will simply compute the CS at all
        times. To compute the CS at every other time, for example,
        simply set times=np.arange(0, N, step=2).


    Returns
    -------
    CIs_lower, list of positive reals
        lower part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    if n is None:
        n = np.ones(len(x))
    # We need x and n to be numpy arrays
    x = np.array(x)
    n = np.array(n)
    
    if times is not None:
        x = np.add.reduceat(x, times)
        n = np.add.reduceat(n, times)

    # cumulative sum
    S_t = np.cumsum(x)
    # intrinsic time
    t = np.cumsum(n)

    # Get x into "overparameterized" form as we usually do with multinomials,
    # for example.
    DM_x = np.vstack((x, n - x))
    
    mart_fn = lambda x, m: DMHG_martingale(np.vstack((x, n-x)),
                                           [BB_alpha, BB_beta],
                                           [int(N*m), N-int(N*m)])
    
    l_01, u_01 =\
        cs_from_martingale_2D(x, N, mart_fn, n=n, alpha=alpha, 
                              search_step=search_step,
                              running_intersection=running_intersection)

    return N*l_01, N*u_01


def repeat(l, r):
    """
    Repeat r times each value of list l.
    """
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def repeat_to(l, r):
    """
    Repeat values in list l so that it has r values
    """
    assert r % len(l) == 0

    return repeat(l, r // len(l))
    

def project_linf(x, y, radius, image_std):
    delta = x - y
    delta = 255 * (delta * torch.Tensor(image_std).view(-1, 1, 1))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / torch.Tensor(image_std).view(-1, 1, 1)
    return y + delta


def roundPixel(x, image_mean, image_std):
    x_pixel = 255 * ((x * torch.Tensor(image_std).view(-1, 1, 1)) + torch.Tensor(image_mean).view(-1, 1, 1))
    y = torch.round(x_pixel).clamp(0, 255)
    y = ((y / 255.0) - torch.Tensor(image_mean).view(-1, 1, 1)) / torch.Tensor(image_std).view(-1, 1, 1)
    return y


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        self._num_updates = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        self._num_updates += 1
        # update learning rate
        new_lr = self.get_lr_for_step(self._num_updates)
        for param_group in self.param_groups:
            param_group['lr'] = new_lr


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    lr_schedule = None
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None or (split[0] == "lr" and "-" in split[1])
            if "lr" in split[0]:
                lr_schedule = [float(lr) for lr in split[1].split("-")]
                optim_params[split[0]] = float(lr_schedule[0])

                # There is no "schedule" if the learning rate stays constant
                if len(lr_schedule) == 1:
                    lr_schedule = None
            else:
                optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'sparseadam':
        optim_fn = optim.SparseAdam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    # print("Schedule of %s: %s" % (s, str(lr_schedule)))

    return optim_fn(parameters, **optim_params), lr_schedule


def de_normalize(datasets, img_mean, img_std):
    """
    De-normalize the dataset.

    Args:
        datasets: The dataset to be de-normalized.
        img_mean: The mean value of the image.
        img_std: The standard deviation of the image.

    Returns:
        The de-normalized dataset.
    """
    for i in range(len(datasets)):
        for j in range(len(datasets[i])):
            img, label = datasets[i][j]
            img = img * torch.Tensor(img_std).view(-1, 1, 1) + torch.Tensor(img_mean).view(-1, 1, 1)
            datasets[i][j] = (img, label)
    return datasets


'''RData'''
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

PIL_INTERPOLATION = {
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING
}

class DifferentiableDataAugmentation:
    def __init__(self):
        pass

    def sample_params(self, x, seed=None):
        # Sample parameters for a given data augmentation
        return 0

    def apply(self, x, params):
        # Apply data augmentation to image
        assert params == 0

        return x

    def __call__(self, x, params):
        return self.apply(x, params)


class CenterCrop(DifferentiableDataAugmentation):

    def __init__(self, resize, crop_size, interpolation='bilinear'):
        assert resize > crop_size
        self.resize = resize
        self.crop_size = crop_size
        self.half_size = crop_size // 2
        self.interpolation = interpolation

    def sample_params(self, x, seed=None):
        return 0

    def apply(self, x, augmentation):
        assert augmentation == 0

        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            min_dim = min(x.size()[2:])
            scale = self.resize / min_dim

            x_resized = F.interpolate(x, scale_factor=scale, mode=self.interpolation)
            x_resized = x_resized.clamp(min=x.min().item(), max=x.max().item())

            i_center, j_center = x_resized.size(2) // 2, x_resized.size(3) // 2

            return x_resized[..., i_center - self.half_size:i_center + self.half_size, j_center - self.half_size:j_center + self.half_size]
        else:
            x = F_img.resize(x, self.resize, PIL_INTERPOLATION[self.interpolation])

            return F_img.center_crop(x, self.crop_size)


class RandomResizedCropFlip(DifferentiableDataAugmentation):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), flip=True, interpolation='bilinear'):
        assert len(ratio) == 2
        assert len(scale) == 2

        self.ratio = ratio
        self.scale = scale
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.flip = flip
        self.interpolation = interpolation


    def sample_params(self, x, seed: int=None):
        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            width, height = x.size(3), x.size(2)
        elif type(x) is Image.Image:
            width, height = x.size

        if seed is not None:
            random.seed(seed)

        flip = random.randint(0, 1) if self.flip else 0
        area = width * height

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, flip

        # Fallback to central crop
        in_ratio = width / height
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w, flip


    def apply(self, x, augmentation):
        i, j, h, w, flip = augmentation

        if type(x) is torch.Tensor:
            assert len(x.size()) == 4
            x_resized = F.interpolate(x[..., i:(i+h), j:(j+w)], size=self.size, mode=self.interpolation)
            x_resized = x_resized.clamp(min=x.min().item(), max=x.max().item())

            if flip:
                x_resized = x_resized[..., torch.arange(x_resized.size(-1) - 1, -1, -1)]
        else:
            x_resized = F_img.resized_crop(x, i, j, h, w, self.size, PIL_INTERPOLATION[self.interpolation])
            if flip:
                x_resized = F_img.hflip(x_resized)

        return x_resized

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Initialation
        """
        self.data = data
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = None
    def __len__(self):
        """
        Return the size
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the sample (img, label)
        """
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label


class Circle_Dataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Initialation
        """
        self.data = data
        self.transform = transform if transform is not None else lambda x: x

    def __len__(self):
        """
        Return the size
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the sample (img, label)
        """
        idx = idx % len(self.data)
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label


class RandomTransform:
    """Crop the given tensor at a random location."""

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud

        self.mode = mode
        self.align = align

    @staticmethod
    def build_grid(source_size, target_size):
        """Build the grid for random cropping."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """Generate a random crop grid."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)
        
        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def __call__(self, x, randgen=None):
        """Apply the random crop."""
        # Make a random shift grid for each batch
        x = x.unsqueeze(0)
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode).squeeze(0)


class Normalized_to_training_transform:
    def __init__(self, dataset='cifar10', img_size=256, crop_size=224, transform=None, transform_type='center'):
        self.mean = imagenet_mean
        self.std = imagenet_std
        if dataset == 'cifar10':
            self.mean = cifar10_mean
            self.std = cifar10_std
        elif dataset == 'cifar100':
            self.mean = cifar100_mean
            self.std = cifar100_std
        elif dataset == 'mnist':
            self.mean = mnist_mean
            self.std = mnist_std
        elif dataset == 'imagenet':
            self.mean = imagenet_mean
            self.std = imagenet_std
        elif dataset == 'tiny_imagenet':
            self.mean = tiny_imagenet_mean
            self.std = tiny_imagenet_std

        if transform is not None:
            self.transform = transform
        else:
            if dataset in ["imagenet", "flickr", "cub", "places205", "imagenet1k"]:
                if transform_type == "random":
                    self.transform = transforms.Compose([
                        transforms.RandomResizedCrop(crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean, std=self.std),
                    ])
                elif transform_type == "center":
                    self.transform = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean, std=self.std),
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])

    def __call__(self, img):
        # input img is Normalized tensor
        img = self.denormalize(img)
        img = transforms.ToPILImage()(img)
        img = self.transform(img)
        return img

    def denormalize(self, img):
        """
        input img is Normalized tensor, 
        and this operation is denormalization
        """
        img = img.mul_(torch.tensor(self.std).view(-1,1,1)).add_(torch.tensor(self.mean).view(-1,1,1))
        return img