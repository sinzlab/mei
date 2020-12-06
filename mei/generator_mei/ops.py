import warnings

import torch
from torch import nn
import torch.nn.functional as F
from scipy import signal
import inspect
import functools


def varargin(f):
    """Decorator to make a function able to ignore named parameters not declared in its
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

    # Find the name of parameters expected by f
    f_params = inspect.signature(f).parameters.values()
    param_names = [p.name for p in f_params]  # name of parameters expected by f
    receives_kwargs = any(
        [p.kind == inspect.Parameter.VAR_KEYWORD for p in f_params]
    )  # f receives a dictionary of **kwargs

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not receives_kwargs:
            # Ignore named parameters not expected by f
            kwargs = {k: kwargs[k] for k in kwargs.keys() if k in param_names}
        return f(*args, **kwargs)

    return wrapper


class Compose(nn.Module):
    """Chain a set of operations into a single function.

    Each function must receive one positional argument and any number of keyword
    arguments. Each function is called with the output of the previous one (as its
    positional argument) and all keyword arguments of the original call.

    Arguments:
        operations (list): List of operations.
    """

    def __init__(self, operations, defaults_to_none=True):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.defaults_to_none = defaults_to_none

    def forward(self, x, **kwargs):
        # if no operations give and defaults_to_none, return None
        if len(self.operations) == 0 and self.defaults_to_none:
            return None

        out = x
        for op in self.operations:
            out = op(out, **kwargs)

        return out

    def __getitem__(self, item):
        return self.operations[item]


class Combine(nn.Module):
    """Applies different operations to an input and combines its output.

    Arguments:
        operations (list): List of operations
        combine_op (function): Function used to combine the results of all the operations.
    """

    def __init__(self, operations, combine_op=torch.sum):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.combine_op = combine_op

    def forward(self, *args, **kwargs):
        if len(self.operations) == 0:
            if self.defaults_to_none:
                return
        else:
            results = [op(*args, **kwargs) for op in self.operations]
            return self.combine_op(torch.stack(results, dim=0))

    def __getitem__(self, item):
        return self.operations[item]


################################## REGULARIZERS ##########################################
class TotalVariation(nn.Module):
    """Total variation regularization.

    Arguments:
        weight (float): Weight of the regularization.
        isotropic (bool): Whether to use the isotropic or anisotropic definition of Total
            Variation. Default is anisotropic (l1-norm of the gradient).
    """

    def __init__(self, weight=1, isotropic=False):
        super().__init__()
        self.weight = weight
        self.isotropic = isotropic

    @varargin
    def forward(self, x):
        # Using the definitions from Wikipedia.
        diffs_y = torch.abs(x[:, :, 1:] - x[:, :, -1:])
        diffs_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        if self.isotropic:
            tv = (
                torch.sqrt(diffs_y[:, :, :, :-1] ** 2 + diffs_x[:, :, :-1, :] ** 2)
                .reshape(len(x), -1)
                .sum(-1)
            )  # per image
        else:
            tv = diffs_y.reshape(len(x), -1).sum(-1) + diffs_x.reshape(len(x), -1).sum(
                -1
            )  # per image
        loss = self.weight * torch.mean(tv)

        return loss


class LpNorm(nn.Module):
    """Computes the lp-norm of an input.

    Arguments:
        weight (float): Weight of the regularization
        p (int): Degree for the l-p norm.
    """

    def __init__(self, weight=1, p=6):
        super().__init__()
        self.weight = weight
        self.p = p

    @varargin
    def forward(self, x):
        lpnorm = (torch.abs(x) ** self.p).reshape(len(x), -1).sum(-1) ** (1 / self.p)
        loss = self.weight * torch.mean(lpnorm)
        return loss


class Similarity(nn.Module):
    """Compute similarity metrics across all examples in one batch.

    Arguments:
        weight (float): Weight of the regularization.
        metric (str): What metric to use when computing pairwise similarities. One of:
            correlation: Masked correlation.
            cosine: Cosine similarity of the masked input.
            neg_euclidean: Negative of euclidean distance between the masked input.
        combine_op (function): Function used to agglomerate pairwise similarities.
        mask (torch.tensor or None): Mask to use when calculating similarities. Expected
            to be in [0, 1] range and be broadcastable with input.
    """

    def __init__(self, weight=1, metric="correlation", combine_op=torch.max, mask=None):
        super().__init__()
        self.weight = weight
        self.metric = metric
        self.combine_op = combine_op
        if mask is None:
            self.mask = None
        else:
            self.register_buffer("mask", mask)

    @varargin
    def forward(self, x):
        if len(x) < 2:
            warnings.warn(
                "Only one image in the batch. Similarity regularization will" "return 0"
            )
            return 0

        # Mask x
        masked_x = x if self.mask is None else x * self.mask
        flat_x = masked_x.view(len(x), -1)

        # Compute similarity matrix
        if self.metric == "correlation":
            if self.mask is None:
                residuals = flat_x - flat_x.mean(-1, keepdim=True)
                numer = residuals @ residuals.t()
                ssr = (residuals ** 2).sum(-1)
            else:
                mask_sum = self.mask.sum() * (
                    flat_x.shape[-1] / len(self.mask.view(-1))
                )
                mean = flat_x.sum(-1) / mask_sum
                residuals = x - mean.view(len(x), *[1] * (x.dim() - 1))  # N x 1 x 1 x 1
                numer = (
                    (residuals[None, :] * residuals[:, None] * self.mask)
                    .view(len(x), len(x), -1)
                    .sum(-1)
                )
                ssr = ((residuals ** 2) * self.mask).view(len(x), -1).sum(-1)
            sim_matrix = numer / (torch.sqrt(torch.ger(ssr, ssr)) + 1e-9)
        elif self.metric == "cosine":
            norms = torch.norm(flat_x, dim=-1)
            sim_matrix = flat_x @ flat_x.t() / (torch.ger(norms, norms) + 1e-9)
        elif self.metric == "neg_euclidean":
            sim_matrix = -torch.norm(flat_x[None, :, :] - flat_x[:, None, :], dim=-1)
        else:
            raise ValueError("Invalid metric name:{}".format(self.metric))

        # Compute overall similarity
        triu_idx = torch.triu(torch.ones(len(x), len(x)), diagonal=1) == 1
        similarity = self.combine_op(sim_matrix[triu_idx])

        return self.weight * similarity


################################ TRANSFORMS ##############################################
class Jitter(nn.Module):
    """Jitter the image at random by some certain amount.

    Arguments:
        max_jitter(tuple of ints): Maximum amount of jitter in y, x.
    """

    def __init__(self, max_jitter):
        super().__init__()
        self.max_jitter = (
            max_jitter if isinstance(max_jitter, tuple) else (max_jitter, max_jitter)
        )

    @varargin
    def forward(self, x):
        # Sample how much to jitter
        jitter_y = torch.randint(
            -self.max_jitter[0], self.max_jitter[0] + 1, (1,), dtype=torch.int32
        ).item()
        jitter_x = torch.randint(
            -self.max_jitter[1], self.max_jitter[1] + 1, (1,), dtype=torch.int32
        ).item()

        # Pad and crop the rest
        pad_y = (jitter_y, 0) if jitter_y >= 0 else (0, -jitter_y)
        pad_x = (jitter_x, 0) if jitter_x >= 0 else (0, -jitter_x)
        padded_x = F.pad(x, pad=(*pad_x, *pad_y), mode="reflect")

        # Crop
        h, w = x.shape[-2:]
        jittered_x = padded_x[
            ...,
            slice(0, h) if jitter_y > 0 else slice(-jitter_y, None),
            slice(0, w) if jitter_x > 0 else slice(-jitter_x, None),
        ]

        return jittered_x


class RandomCrop(nn.Module):
    """Take a random crop of the input image.

    Arguments:
        height (int): Height of the crop.
        width (int): Width of the crop
    """

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    @varargin
    def forward(self, x):
        crop_y = torch.randint(
            0, max(0, x.shape[-2] - self.height) + 1, (1,), dtype=torch.int32
        ).item()
        crop_x = torch.randint(
            0, max(0, x.shape[-1] - self.width) + 1, (1,), dtype=torch.int32
        ).item()
        cropped_x = x[..., crop_y : crop_y + self.height, crop_x : crop_x + self.width]

        return cropped_x


class BatchedCrops(nn.Module):
    """Create a batch of crops of the original image.

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
        super().__init__()
        self.height = height
        self.width = width
        self.step_size = step_size if isinstance(step_size, tuple) else (step_size,) * 2
        self.sigma = (
            sigma if sigma is None or isinstance(sigma, tuple) else (sigma,) * 2
        )

        # If needed, create gaussian mask
        if sigma is not None:
            y_gaussian = signal.gaussian(height, std=self.sigma[0])
            x_gaussian = signal.gaussian(width, std=self.sigma[1])
            self.mask = y_gaussian[:, None] * x_gaussian

    @varargin
    def __call__(self, x):
        if len(x) > 1:
            raise ValueError("x can only have one example.")
        if x.shape[-2] < self.height or x.shape[-1] < self.width:
            raise ValueError("x should be larger than the expected crop")

        # Take crops
        crops = []
        for i in range(0, x.shape[-2] - self.height + 1, self.step_size[0]):
            for j in range(0, x.shape[-1] - self.width + 1, self.step_size[1]):
                crops.append(x[..., i : i + self.height, j : j + self.width])
        crops = torch.cat(crops, dim=0)

        # Multiply by a gaussian mask if needed
        if self.sigma is not None:
            mask = torch.as_tensor(self.mask, device=crops.device, dtype=crops.dtype)
            crops = crops * mask

        return crops


class ChangeRange(nn.Module):
    """This changes the range of x as follows:
        new_x = sigmoid(x) * (desired_max - desired_min) + desired_min

    Arguments:
        x_min (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
        x_max (float or tensor): Minimum desired value for the output. If a tensor it
            needs to be broadcastable with x.
    """

    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def forward(self, x):
        new_x = torch.sigmoid(x) * (self.x_max - self.x_min) + self.x_min
        return new_x


class Resize(nn.Module):
    """Resize images.

    Arguments:
        scale_factor (float or tuple): Factors to rescale the images:
            new_h, new_w = round(scale_factor * (old_h, old_w)).
        resize_method (str): 'nearest' or 'bilinear' interpolation.

    Note:
        This changes the dimensions of the image.
    """

    def __init__(self, scale_factor, resize_method="bilinear"):
        super().__init__()
        self.scale_factor = (
            scale_factor
            if isinstance(scale_factor, tuple)
            else (scale_factor, scale_factor)
        )
        self.resample_method = resize_method

    @varargin
    def forward(self, x):
        new_height = int(round(x.shape[-2] * self.scale_factor[0]))
        new_width = int(round(x.shape[-1] * self.scale_factor[0]))
        return F.upsample(x, (new_height, new_width), mode=self.resize_method)


class GrayscaleToRGB(nn.Module):
    """ Transforms a single channel image into three channels (by copying the channel)."""

    @varargin
    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError("Image is not grayscale!")

        return x.expand(-1, 3, -1, -1)


class Identity(nn.Module):
    """ Transform that returns the input as is."""

    @varargin
    def forward(self, x):
        return x


############################## GRADIENT OPERATIONS #######################################
class ChangeNorm(nn.Module):
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, mu=0):
        super().__init__()
        self.norm = torch.tensor(norm)
        self.mu = mu

    @varargin
    def forward(self, x):
        x_norm = torch.norm((x - self.mu).view(len(x), -1), dim=-1)
        renorm = (
            x * (self.norm.to(x.device) / x_norm).view(len(x), *[1] * (x.dim() - 1))
            + self.mu
        )
        return renorm


class ScaleNorm(nn.Module):
    """Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, mu=0, dims=None):
        super().__init__()
        self.norm = torch.tensor(norm)
        self.mu = torch.tensor(mu)

    @varargin
    def forward(self, x):
        x_norm = torch.norm((x - self.mu).view(len(x), -1), dim=-1)
        renorm = (
            x * (self.norm.to(x.device) / x_norm).view(len(x), *[1] * (x.dim() - 1))
            + self.mu
        )
        return renorm


class ClipRange(nn.Module):
    """Clip the value of x to some specified range.

    Arguments:
        x_min (float): Lower valid value.
        x_max (float): Higher valid value.
    """

    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def forward(self, x):
        return torch.clamp(x, self.x_min, self.x_max)


class FourierSmoothing(nn.Module):
    """Smooth the input in the frequency domain.

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
        super().__init__()
        self.freq_exp = freq_exp

    @varargin
    def __call__(self, x):
        # Create mask of frequencies (following np.fft.rfftfreq and np.fft.fftfreq docs)
        h, w = x.shape[-2:]
        freq_y = (
            torch.cat(
                [
                    torch.arange((h - 1) // 2 + 1, dtype=torch.float32),
                    -torch.arange(h // 2, 0, -1, dtype=torch.float32),
                ]
            )
            / h
        )  # fftfreq
        freq_x = torch.arange(w // 2 + 1, dtype=torch.float32) / w  # rfftfreq
        yx_freq = torch.sqrt(freq_y[:, None] ** 2 + freq_x ** 2)

        # Create smoothing mask
        norm_freq = yx_freq * torch.sqrt(torch.tensor(2.0))  # 0-1
        mask = (1 - norm_freq) ** self.freq_exp

        # Smooth
        freq = torch.rfft(x, signal_ndim=2)  # same output as np.fft.rfft2
        mask = torch.as_tensor(mask, device=freq.device, dtype=freq.dtype).unsqueeze(-1)
        smooth = torch.irfft(freq * mask, signal_ndim=2, signal_sizes=x.shape[-2:])
        return smooth


class DivideByMeanOfAbsolute(nn.Module):
    """ Divides x by the mean of absolute x. """

    @varargin
    def forward(self, x):
        return x / torch.abs(x).mean(axis=tuple(range(1, x.dim())), keepdim=True)


class MultiplyBy(nn.Module):
    """Multiply x by some constant.

    Arguments:
        const: Number x will be multiplied by
        decay_factor: Compute const every iteration as `const + decay_factor * (iteration
            - 1)`. Ignored if None.
    """

    def __init__(self, const, decay_factor=None):
        super().__init__()
        self.const = const
        self.decay_factor = decay_factor

    @varargin
    def forward(self, x, iteration=None):
        if self.decay_factor is None:
            const = self.const
        else:
            const = self.const + self.decay_factor * (iteration - 1)

        return const * x


########################### POST UPDATE OPERATIONS #######################################
class GaussianBlur(nn.Module):
    """Blur an image with a Gaussian window.

    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
    """

    def __init__(self, sigma, decay_factor=None, truncate=4, pad_mode="reflect"):
        super().__init__()
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode

    @varargin
    def forward(self, x, iteration=None):
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
        padded_x = F.pad(
            x, pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize), mode=self.pad_mode
        )
        blurred_x = F.conv2d(
            padded_x,
            y_gaussian.repeat(num_channels, 1, 1)[..., None],
            groups=num_channels,
        )
        blurred_x = F.conv2d(
            blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels
        )
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize

        return final_x


class ChangeStd(nn.Module):
    """Change the standard deviation of input.

    Arguments:
        std (float or tensor): Desired std. If tensor, it should be the same length as x.
        preserve_mean (bool, optional): If True, preserves the mean. Defaults to True.

    """

    def __init__(self, std, preserve_mean=True):
        super().__init__()
        self.std = std
        self.preserve_mean = preserve_mean

    @varargin
    def forward(self, x):
        x_std = torch.std(x.view(len(x), -1), dim=-1)
        if self.preserve_mean:
            x_mu = x.mean(dims=tuple(range(x.dims())), keepdim=True)
        else:
            x_mu = 0

        fixed_std = (x - x_mu) * (self.std / (x_std + 1e-9)).view(
            len(x), *[1] * (x.dim() - 1)
        ) + x_mu
        return fixed_std
