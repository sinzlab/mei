import numpy as np
import torch


def create_gabor(height, width, orientation, phase, wavelength, sigma, dx=0, dy=0):
    """Create a Gabor patch that will let the gradient flow back to its parameters.

    Arguments:
        height (int): Height of the gabor in pixels.
        width (int): Width of the gabor in pixels.
        orientation (scalar FloatTensor): Orientation of the gabor in radians.
        phase (scalar FloatTensor): Phase of the gabor patch in radians.
        wavelength (scalar FloatTensor): Wavelength of the gabor expressed as a proportion
            of height.
        sigma (scalar FloatTensor): Standard deviation of the gaussian window expressed as
            a proportion of height.
        dx (scalar FloatTensor): Amount of shifting in x (expressed as a proportion of
            width) [-0.5, 0.5]. Positive moves the gabor to the right.
        dy (scalar FloatTensor): Amount of shifting in y (expressed as a  proportion of
            height) [-0.5, 0.5]. Positive moves the gabor down.

    Returns:
        A Gabor patch (torch.FloatTensor) with the desired properties.

    Note:
        This diverges from the Gabor formulation in https://en.wikipedia.org/wiki/Gabor_filter:
        * theta is the orientation of the gabor rather than "the orientation of the normal
            to the parallel stripes" (i.e., theta = wikipedia_theta - pi/2).
        * rotations are clockwise (i.e theta = - wikipedia_theta).
        * for some dx, dy, the gaussian mask will always be in the same place regardless
            of orientation.
        * sigma and wavelength are expressed as proportions of height rather than width.
    """
    # Basic checks
    # orientation = torch.remainder(orientation, np.pi) # 0-180
    # phase = torch.remainder(phase, 2 * np.pi) # 0-360
    if wavelength <= 0:
        raise ValueError('wavelength needs to be positive')
    if sigma <= 0:
        raise ValueError('sigma needs to be positive')

    # Create grid
    y = np.arange(height) + 0.5  # in pixels : 0.5, 1.5, ... w-1.5, w-0.5
    y = (y - height / 2) / height  # from -0.5 to 0.5
    x = np.arange(width) + 0.5  # in pixels : 0.5, 1.5, ... h-1.5, h-0.5
    x = (x - width / 2) / height  # from -(w / 2h) to (w / 2h)
    yp, xp = np.meshgrid(y, x, indexing='ij')
    yp = orientation.new_tensor(yp)
    xp = orientation.new_tensor(xp)

    # Sample gaussian mask
    dx = dx * width / height  # re-express as a proportion of height
    gauss = torch.exp(-((xp - dx) ** 2 + (yp - dy) ** 2) / (2 * sigma ** 2))

    # Sample sinusoid
    x_rot = xp * torch.sin(orientation) + yp * torch.cos(orientation)
    sin = torch.cos((2 * np.pi / wavelength) * x_rot + phase)

    # Create gabor
    gabor = gauss * sin

    return gabor


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