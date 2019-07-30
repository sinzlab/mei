""" Code to reproduce some standard feature visualizations.

Ref: Olah, et al., "Feature Visualization", Distill, 2017.
"""
import warnings

import torch
import torch.nn.functional as F
from scipy import signal
from torch import optim


class FeatureVisException(Exception):
    """ Catch-all exception for errors generated during feature visualization."""
    pass


def gradient_ascent(f, x, transform=None, regularization=None, gradient_f=None,
                    post_update=None, optim_name='SGD', step_size=0.1, optim_kwargs={},
                    num_iterations=1000, save_iters=None, print_iters=100):
    """ Maximize f(x) via gradient ascent.

    Objective: f(transform(x)) - regularization(transform(x))
    Update: x_{t+1} = post_update(x_{t} + step_size * gradient_f(x_{t}.grad))

    Arguments:
        f (function): Real-valued differentiable function to be optimized
        x (torch.Tensor): Initial guess of the input to optimize.
        transform (function): Differentiable transformation applied to x before sending it
            through the model, e.g., an image generator, jittering, scaling, etc.
        regularization (function): Differentiable regularization term, e.g., natural
            prior, total variation, bilateral filters, etc.
        gradient_f (function): Non-differentiable. Receives the gradient of x and outputs
            a preconditioned gradient, e.g., blurring, masking, etc.
        post_update (function): Non-differentiable. Function applied to x after each
            gradient update, e.g., keep the image norm to some value, blurring, etc.
        optim_name (string): Optimizer to use: SGD or Adam.
        step_size (float): Size of the step size to give every iteration.
        optim_kwargs (dict): Dictionary with kwargs for the optimizer
        num_iterations (int): Number of gradient ascent steps.
        save_iters (None or int): How often to save x. If None, it returns the best x;
            otherwise it saves x after each save_iters iterations.
        print_iters (int): Print some results every print_iters iterations.

    Returns:
        optimal_x (torch.Tensor): x that maximizes the desired function. If save_iters is
            not None, this will be a list of tensors.
        fevals (list): Function evaluations at each iteration. We also evaluate at x_0
            (the original input) so this will have max_iterations + 1 elements.
        reg_terms (list): Value of the regularization term at each iteration. We also
            evaluate at x_0 (the original input) so this will have max_iterations + 1
            elements. Empty if regularization is None.

    Note:
        transform, regularization, gradient_f and post_update receive one positional
        parameter (its input) and the following optional named parameters:
            iteration (int): Current iteration (starts at 1).

        The number of optional parameters may increase so we recommend to write functions
        that receive **kwargs (or use the varargin decorator below) to make sure they will
        still work if we add other optional parameters in the future.
    """
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError('x must be of torch.float32 dtype')
    x = x.detach().clone()  # to avoid changing original
    x.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store xs (ignored if save_iters is None)
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Transform input
        transformed_x = x if transform is None else transform(x, iteration=i)

        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())

        # Regularization
        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i)
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x.grad is None:
            raise FeatureVisException('Gradient did not reach x.')

        # Precondition gradient
        x.grad = x.grad if gradient_f is None else gradient_f(x.grad, iteration=i)
        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn('Gradient for x is all zero')

        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))

        # Save x
        if save_iters is not None and i % save_iters == 0:
            saved_xs.append(x.detach().clone())

    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x if transform is None else transform(x, iteration=i + 1)

        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i + 1)
            reg_terms.append(reg_term.item())
    print('Final f(x) = {:.2f}'.format(fevals[-1]))

    # Set opt_x
    opt_x = x.detach().clone() if save_iters is None else saved_xs

    return opt_x, fevals, reg_terms


############################### UTILITY CLASSES ########################################
def varargin(f):
    """ Decorator to make a function able to ignore named parameters not declared in its
     definition.

    Arguments:
        f (function): Original function.

    Usage:
            @varargin
            def my_f(x):
                # ...
        is equivalent to
            def my_f(x, **kwargs):
                #...
        Using the decorator is recommended because it makes it explicit that my_f won't
        use arguments received in the kwargs dictionary.
    """
    import inspect
    import functools

    # Find the name of parameters expected by f
    f_params = inspect.signature(f).parameters.values()
    param_names = [p.name for p in f_params]  # name of parameters expected by f
    receives_kwargs = any([p.kind == inspect.Parameter.VAR_KEYWORD for p in
                           f_params])  # f receives a dictionary of **kwargs
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not receives_kwargs:
            # Ignore named parameters not expected by f
            kwargs = {k: kwargs[k] for k in kwargs.keys() if k in param_names}
        return f(*args, **kwargs)

    return wrapper


class Compose():
    """ Chain a set of operations into a single function.

    Each function must receive one positional argument and any number of keyword
    arguments. Each function is called with the output of the previous one (as its
    positional argument) and all keyword arguments of the original call.

    Arguments:
        operations (list): List of functions.
    """

    def __init__(self, operations):
        self.operations = operations

    def __call__(self, x, **kwargs):
        if len(self.operations) == 0:
            out = None
        else:
            out = self.operations[0](x, **kwargs)
            for op in self.operations[1:]:
                out = op(out, **kwargs)

        return out

    def __getitem__(self, item):
        return self.operations[item]


class Combine():
    """ Applies different operations to an input and combines its output.

    Arguments:
        operations (list): List of operations
        combine_op (function): Function used to combine the results of all the operations.
    """

    def __init__(self, operations, combine_op=torch.sum):
        self.operations = operations
        self.combine_op = combine_op

    def __call__(self, *args, **kwargs):
        if len(self.operations) == 0:
            return
        else:
            results = [op(*args, **kwargs) for op in self.operations]
            return self.combine_op(torch.stack(results, dim=0))

    def __getitem__(self, item):
        return self.operations[item]


################################### MODELS ##############################################
class Ensemble():
    """ Average the response across a set of models.

    Arguments:
        models (list): A list of pytorch models.
        readout_key (str): String identifying the scan whose neurons will be outputted by
            the model
        eye_pos (torch.Tensor): A [1 x 2] tensor with the position of the pupil(x, y).
            This shifts the FOV of all cells. Default (None) is position at center of
            screen (i.e., it disables the shifter).
        behavior (torch.Tensor): A [1 x 3] tensor with the behavior parameters
            (pupil_dilation, dpupil_dilation/dt, treadmill). Default is to return results
            without modulation.
        neuron_idx (int or slice or list): Neuron(s) to return. Default returns all cells.
        y_shift, x_shift (float or torch tensor): Overwrite the learnt (per-cell readout)
            shift with these values for all cells. Values are clipped to [-1, 1] (see
            torch.nn.functional.grid_sample). Default uses the learnt readout shift.
        device (torch.Device or str): Where to load the models.

    Note:
        We copy the models to avoid overwriting the gradients (and grid if x_shift or
        y_shift is set) of the original models. You can access our copy of the models as
        my_ensemble.models.
    """
    def __init__(self, models, readout_key, eye_pos=None, behavior=None,
                 neuron_idx=slice(None), y_shift=None, x_shift=None, device='cuda'):
        import copy

        self.models = [copy.deepcopy(m) for m in models]
        self.readout_key = readout_key
        self.eye_pos = None if eye_pos is None else eye_pos.to(device)
        self.behavior = None if behavior is None else behavior.to(device)
        self.neuron_idx = neuron_idx
        self.y_shift = y_shift
        self.x_shift = x_shift
        self.device = device

        for m in self.models:
            # If needed, change readout shifts to the desired ones
            with torch.no_grad():
                if self.y_shift is not None:
                    m.readout[readout_key].grid[..., 1] = self.y_shift
                if self.x_shift is not None:
                    m.readout[readout_key].grid[..., 0] = self.x_shift

            m.to(device)
            m.eval()

    def __call__(self, x):
        resps = [m(x, self.readout_key, eye_pos=self.eye_pos, behavior=self.behavior)[:,
                 self.neuron_idx] for m in self.models]  # num_models x batch_size x num_neurons
        resp = torch.stack(resps).mean(0).mean(0)  # num_neurons

        return resp


class VGG19Core():
    """ A pretrained VGG-19. Output will be intermediate feature representation
    (N x C x H x W) at the desired layer.

    Arguments:
        layer (int): Index (0-based) of the layer that will be optimized.
        use_batchnorm (boolean): Whether to download the version with batchnorm.
        device (torch.Device or str): Where to place the model.
    """
    def __init__(self, layer, use_batchnorm=True, device='cuda'):
        from torchvision import models

        vgg19 = (models.vgg19_bn(pretrained=True) if use_batchnorm else
                 models.vgg19(pretrained=True))
        if layer < len(vgg19.features):
            self.model = vgg19.features[:layer + 1]
        else:
            raise ValueError('layer out of range (max is', len(vgg19.features))
        self.model.to(device)
        self.model.eval()

    @varargin
    def __call__(self, x):
        return self.model(x)


class VGG19():
    """ A pretrained VGG-19. Output will be the average of one channel across spatial
    dimensions.

    Arguments:
        layer (int): Index (0-based) of the layer that will be optimized.
        channel (int)_: Index (0-based) of the channel that will be optimized.
        use_batchnorm (boolean): Whether to download the version with batchnorm.
        device (torch.Device or str): Where to place the model.
    """
    def __init__(self, layer, channel, use_batchnorm=True, device='cuda'):
        self.model = VGG19Core(layer, use_batchnorm, device)
        self.channel = channel

    def __call__(self, x):
        resp = self.model(x)[:, self.channel, :, :].mean()
        return resp


################################## REGULARIZERS ##########################################
class TotalVariation():
    """ Total variation regularization.

    Arguments:
        weight (float): Weight of the regularization.
        isotropic (bool): Whether to use the isotropic or anisotropic definition of Total
            Variation. Default is anisotropic (l1-norm of the gradient).
    """
    def __init__(self, weight=1, isotropic=False):
        self.weight = weight
        self.isotropic = isotropic

    @varargin
    def __call__(self, x):
        # Using the definitions from Wikipedia.
        diffs_y = torch.abs(x[:, :, 1:] - x[:, :, -1:])
        diffs_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        if self.isotropic:
            tv = torch.sqrt(diffs_y[:, :, :, :-1] ** 2 +
                            diffs_x[:, :, :-1, :] ** 2).reshape(len(x), -1).sum(-1)  # per image
        else:
            tv = diffs_y.reshape(len(x), -1).sum(-1) + diffs_x.reshape(len(x), -1).sum(-1)  # per image
        loss = self.weight * torch.mean(tv)

        return loss


class LpNorm():
    """Computes the lp-norm of an input.

    Arguments:
        weight (float): Weight of the regularization
        p (int): Degree for the l-p norm.
    """
    def __init__(self, weight=1, p=6):
        self.weight = weight
        self.p = p

    @varargin
    def __call__(self, x):
        lpnorm = (torch.abs(x) ** self.p).reshape(len(x), -1).sum(-1) ** (1/self.p)
        loss = self.weight * torch.mean(lpnorm)
        return loss


class Similarity():
    """ Compute similarity metrics across all examples in one batch.

    Arguments:
        weight (float): Weight of the regularization.
        metric (str): What metric to use when computing pairwise similarities. One of:
            correlation: Masked correlation.
            cosine: Cosine similarity of the masked input.
            euclidean: Negative of euclidean distance between the masked input.
        combine_op (function): Function used to agglomerate pairwise similarities.
        mask (torch.tensor or None): Mask to use when calculating similarities. Expected
            to be in [0, 1] range and be broadcastable with input.
    """
    def __init__(self, weight=1, metric='correlation', combine_op=torch.max, mask=None):
        self.weight = weight
        self.metric = metric
        self.combine_op = combine_op
        self.mask = mask

    @varargin
    def __call__(self, x):
        if len(x) < 2:
            warnings.warn('Only one image in the batch. Similarity regularization will'
                          'return 0')
            return 0

        # Mask x
        masked_x = x if self.mask is None else x * self.mask
        flat_x = masked_x.view(len(x), -1)

        # Compute similarity matrix
        if self.metric == 'correlation':
            if self.mask is None:
                residuals = flat_x - flat_x.mean(-1, keepdim=True)
                numer = torch.mm(residuals, residuals.t())
                ssr = (residuals ** 2).sum(-1)
            else:
                mask_sum = self.mask.sum() * (flat_x.shape[-1] / len(self.mask.view(-1)))
                mean = flat_x.sum(-1) / mask_sum
                residuals = x - mean.view(len(x), *[1, ] * (x.dim() - 1))  # N x 1 x 1 x 1
                numer = (residuals[None, :] * residuals[:, None] * self.mask).view(
                    len(x), len(x), -1).sum(-1)
                ssr = ((residuals ** 2) * self.mask).view(len(x), -1).sum(-1)
            sim_matrix = numer / (torch.sqrt(torch.ger(ssr, ssr)) + 1e-9)
        elif self.metric == 'cosine':
            norms = torch.norm(flat_x, dim=-1)
            sim_matrix = torch.mm(flat_x, flat_x.t()) / (torch.ger(norms, norms) + 1e-9)
        elif self.metric == 'euclidean':
            sim_matrix = -torch.norm(flat_x[None, :, :] - flat_x[:, None, :], dim=-1)
        else:
            raise ValueError('Invalid metric name:{}'.format(self.metric))

        # Compute overall similarity
        triu_idx = torch.triu(torch.ones(len(x), len(x)), diagonal=1) == 1
        similarity = self.combine_op(sim_matrix[triu_idx])

        loss = self.weight * similarity

        return loss


# class PixelCNNRegularizer():
#     def __init__(self, weight=1):
#         self.weight = weight
#
#         self.pixel_cnn = ... # load the model
#
#     @varargin
#     def __call__(self, x):
#         # Modify x to make it a valid input to pixel cnn (add channels)
#         prior = self.pixel_cnn(x)
#         loss = self.weight * prior


################################ TRANSFORMS ##############################################
class Jitter():
    """ Jitter the image at random by some certain amount.

    Arguments:
        max_jitter(tuple of ints): Maximum amount of jitter in y, x.
    """
    def __init__(self, max_jitter):
        self.max_jitter = max_jitter if isinstance(max_jitter, tuple) else (max_jitter,
                                                                            max_jitter)
    @varargin
    def __call__(self, x):
        # Sample how much to jitter
        jitter_y = torch.randint(-self.max_jitter[0], self.max_jitter[0] + 1, (1,),
                                 dtype=torch.int32).item()
        jitter_x = torch.randint(-self.max_jitter[1], self.max_jitter[1] + 1, (1,),
                                 dtype=torch.int32).item()

        # Pad and crop the rest
        pad_y = (jitter_y, 0) if jitter_y >= 0 else (0, -jitter_y)
        pad_x = (jitter_x, 0) if jitter_x >= 0 else (0, -jitter_x)
        padded_x = F.pad(x, pad=(*pad_x,  *pad_y), mode='reflect')

        # Crop
        h, w = x.shape[-2:]
        jittered_x = padded_x[..., slice(0, h) if jitter_y > 0 else slice(-jitter_y, None),
                              slice(0, w) if jitter_x > 0 else slice(-jitter_x, None)]

        return jittered_x


class RandomCrop():
    """ Take a random crop of the input image.

    Arguments:
        height (int): Height of the crop.
        width (int): Width of the crop
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @varargin
    def __call__(self, x):
        crop_y = torch.randint(0, max(0, x.shape[-2] - self.height) + 1, (1, ),
                               dtype=torch.int32).item()
        crop_x = torch.randint(0, max(0, x.shape[-1] - self.width) + 1, (1, ),
                               dtype=torch.int32).item()
        cropped_x = x[..., crop_y: crop_y + self.height, crop_x: crop_x + self.width]

        return cropped_x


class BatchedCrops():
    """ Create a batch of crops of the original image.

    Arguments:
        height (int): Height of the crop
        width (int): Width of the crop
        step_size (int or tuple): Number of pixels in y, x to step for each crop.
        sigma (float or tuple): Sigma in y, x for the gaussian mask applied to each batch.
            None to avoid masking

    Note:
        Increasing the stride of every convolution to stride * step_size produces the same
        effect in a much more memory efficient way but it will be architecture dependent
        and may not play nice with the rest of transforms.
    """
    def __init__(self, height, width, step_size, sigma=None):
        self.height = height
        self.width = width
        self.step_size = step_size if isinstance(step_size, tuple) else (step_size, ) * 2
        self.sigma = sigma if sigma is None or isinstance(sigma, tuple) else (sigma, ) * 2

        # If needed, create gaussian mask
        if sigma is not None:
            y_gaussian = signal.gaussian(height, std=self.sigma[0])
            x_gaussian = signal.gaussian(width, std=self.sigma[1])
            self.mask = y_gaussian[:, None] * x_gaussian

    @varargin
    def __call__(self, x):
        if len(x) > 1:
            raise ValueError('x can only have one example.')
        if x.shape[-2] < self.height or x.shape[-1] < self.width:
            raise ValueError('x should be larger than the expected crop')

        # Take crops
        crops = []
        for i in range(0, x.shape[-2] - self.height + 1, self.step_size[0]):
            for j in range(0, x.shape[-1] - self.width + 1, self.step_size[1]):
                crops.append(x[..., i : i + self.height, j: j + self.width])
        crops = torch.cat(crops, dim=0)

        # Multiply by a gaussian mask if needed
        if self.sigma is not None:
            mask = torch.as_tensor(self.mask, device=crops.device, dtype=crops.dtype)
            crops = crops * mask

        return crops


class ChangeRange():
    """ This changes the range of x as follows:
        new_x = sigmoid(x) * desired_max + desired_min

    Arguments:
        x_min (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
        x_max (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
    """
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x):
        new_x = torch.sigmoid(x) * self.x_max + self.x_min
        return new_x


class GaborGenerator():
    """ Generate a Gabor patch based on some input parameters.

    Arguments:
        height: Height of the Gabor patch.
        width: Width of the Gabor patch.
        normalize_image: Whether we normalize the gabor patch before returning.
    """
    def __init__(self, height, width, normalize_image=False):
        self.height = height
        self.width = width
        self.normalize_image = normalize_image

    @varargin
    def __call__(self, x):
        """  Create a Gabor patch with the params in x.

        Arguments:
            x (tuple): Tuple with 4-6 parameters: orientation, phase, wavelength, sigma,
                dx and dy. See utils.create_gabor for details.
        """
        from staticnet_invariance import utils

        gabors = []
        for params in x:
            gabor = utils.create_gabor(self.height, self.width, *params)
            if self.normalize_image:
                gabor = (gabor - gabor.mean()) / (gabor.std() + 1e-9)
            gabors.append(gabor)
        gabors = torch.stack(gabors)[:, None, :, :] # add channel dimension

        return gabors


class Resample():
    """ Resample images.

    Arguments:
        scale_factor (float): Factor to rescale the images:
            new_h, new_w = round(scale_factor * (old_h, old_w)).
        resample_method (str): 'nearest' or 'bilinear' interpolation.

    Note:
        This changes the dimensions of the image.
    """
    def __init__(self, scale_factor, resample_method='bilinear'):
        self.scale_factor = scale_factor
        self.resample_method = resample_method

    @varargin
    def __call__(self, x):
        new_height = int(round(x.shape[-2] * self.scale_factor))
        new_width = int(round(x.shape[-1] * self.scale_factor))
        return F.upsample(x, (new_height, new_width), mode=self.resample_method)


class GrayscaleToRGB():
    """ Transforms a single channel image into three channels (by copying the channel)."""
    @varargin
    def __call__(self, x):
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError('Image is not grayscale!')

        return x.expand(-1, 3, -1, -1)


############################## GRADIENT OPERATIONS #######################################
class ChangeNorm():
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """
    def __init__(self, norm):
        self.norm = norm

    @varargin
    def __call__(self, x):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self.norm / x_norm).view(len(x), *[1, ] * (x.dim() -1))
        return renorm


class ClipRange():
    """Clip the value of x to some specified range.

    Arguments:
        x_min (float): Lower valid value.
        x_max (float): Higher valid value.
    """
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x):
        return torch.clamp(x, self.x_min, self.x_max)


class FourierSmoothing():
    """ Smooth the input in the frequency domain.

    Image is transformed to fourier domain, power densities at i, j are multiplied by
    (1 - ||f||)**freq_exp where ||f|| = sqrt(f_i**2 + f_j**2) and the image is brought
    back to the spatial domain:
        new_x = ifft((1 - freqs) ** freq_exp * fft(x))

    Arguments:
        freq_exp (float): Exponent for the frequency mask. Higher numbers produce more
            smoothing.

    Note:
        Consider just using Gaussian blurring. Faster and easier to explain.
    """
    def __init__(self, freq_exp):
        self.freq_exp = freq_exp

    @varargin
    def __call__(self, x):
        # Create mask of frequencies (following np.fft.rfftfreq and np.fft.fftfreq docs)
        h, w = x.shape[-2:]
        freq_y = torch.cat([torch.arange((h - 1) // 2 + 1, dtype=torch.float32),
                            -torch.arange(h // 2, 0, -1, dtype=torch.float32)]) / h  # fftfreq
        freq_x = torch.arange(w // 2 + 1, dtype=torch.float32) / w  # rfftfreq
        yx_freq = torch.sqrt(freq_y[:, None] ** 2 + freq_x ** 2)

        # Create smoothing mask
        norm_freq = yx_freq * torch.sqrt(torch.tensor(2.0)) # 0-1
        mask = (1 - norm_freq) ** self.freq_exp

        # Smooth
        freq = torch.rfft(x, signal_ndim=2) # same output as np.fft.rfft2
        mask = torch.as_tensor(mask, device=freq.device, dtype=freq.dtype).unsqueeze(-1)
        smooth = torch.irfft(freq * mask, signal_ndim=2, signal_sizes=x.shape[-2:])
        return smooth


class DeepDrawUpdateFactor():
    """ Computing the update factor in deepdraw

    x_new = x_old + step_size * update_factor
    x_new = x_old + step_size * [(step_gain * grad) / (255 * torch.abs(grad).mean())]

    Arguments:
        step_gain (float): Scaling factor for the gradient

    Note:
        Original update is this:
            src.data += (step_size / (torch.abs(grad.data).mean() + eps)) * (step_gain / 255) * grad.data
    """
    def __init__(self, step_gain):
        self.step_gain = step_gain

    @varargin
    def __call__(self, x):
        mean_abs_x = torch.abs(x).view(len(x), -1).mean(-1)
        factor = (self.step_gain * x) / (255 * mean_abs_x[:, None, None, None] + 1e-9)
        return factor


class MultiplyBy():
    """Multiply x by some constant.

    Arguments:
        const: Number x will be multiplied by
        decay_factor: Compute const every iteration as `const + decay_factor * (iteration
            - 1)`. Ignored if None.
    """
    def __init__(self, const, decay_factor=None):
        self.const = const
        self.decay_factor = decay_factor

    @varargin
    def __call__(self, x, iteration=None):
        if self.decay_factor is None:
            const = self.const
        else:
            const = self.const + self.decay_factor * (iteration - 1)

        return const * x


########################### POST UPDATE OPERATIONS #######################################
class GaussianBlur():
    """ Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """
    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode='reflect'):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma, ) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode

    @varargin
    def __call__(self, x, iteration=None):
        num_channels = x.shape[1]

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        padded_x = F.pad(x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                         mode=self.pad_mode)
        blurred_x = F.conv2d(padded_x, y_gaussian.repeat(num_channels, 1, 1)[..., None],
                             groups=num_channels)
        blurred_x = F.conv2d(blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1),
                             groups=num_channels)
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum()) # normalize

        return final_x


class ChangeStd():
    """ Change the standard deviation of input.

        Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
    """
    def __init__(self, std):
        self.std = std

    @varargin
    def __call__(self, x):
        x_std= torch.std(x.view(len(x), -1), dim=-1)
        fixed_std = x * (self.std / (x_std + 1e-9)).view(len(x), *[1, ] * (x.dim() -1))
        return fixed_std