import torch


class PNormConstraint:
    """
    Enforces a p-norm constraint on a given image tensor.

    This class is designed to be used as a callable for applying p-norm normalization on an tensor. It ensures
    that the p-norm of the tensor does not exceed a specified maximum value. If the p-norm of the input tensor exceeds
    this value, the tensor is scaled down to meet the constraint.

    Attributes:
        p (int): The order of the norm. A positive integer that determines the type of norm to apply.
        max_value (float): The maximum allowed value for the p-norm of the image.

    Methods:
        __call__(img: torch.Tensor, iteration): Applies the p-norm constraint to the given tensor.

    Parameters:
        norm_value (float): The maximum allowed value for the p-norm of the image.
        p (int, optional): The order of the norm. Defaults to 1, which corresponds to the L1 norm.

    Returns:
        torch.Tensor: The normalized tensor that satisfies the p-norm constraint.
    """

    def __init__(self, norm_value: float, p: int = 1):
        self.p = p
        self.max_value = norm_value

    def __call__(self, img: torch.Tensor, iteration):
        norm_value = torch.norm(img, self.p)
        if norm_value > self.max_value:
            normalized_img = img * (self.max_value / (norm_value + 1e-12))
        else:
            normalized_img = img
        return normalized_img


class PNormConstraintAndClip:
    """
    Enforces a p-norm constraint and pixel value clipping on a given image tensor.

    This class combines p-norm normalization and pixel value clipping to ensure that the p-norm of an image tensor does
     not exceed a specified maximum value, and that each pixel's value remains within specified bounds.

    Attributes:
        max_value (float): The maximum allowed value for the p-norm of the image.
        max_pixel_value (float): The upper limit for pixel values in the image.
        min_pixel_value (float): The lower limit for pixel values in the image.
        p (int): The order of the norm, determining the type of normalization applied.

    Methods:
        __call__(img: torch.Tensor, iteration=None): Applies the p-norm constraint and clips the pixel values
        of the image tensor. It first checks and scales the image tensor if its p-norm exceeds `max_value`.
        Then, it clips the pixel values to ensure they stay within `[min_pixel_value, max_pixel_value]`.

    Parameters:
        norm_value (float, optional): The maximum allowed value for the p-norm of the image. Defaults to 30.0.
        p (int, optional): The order of the norm, with `1` indicating the L1 norm. Defaults to 1.
        max_pixel_value (float, optional): The maximum allowable pixel value in the image. Defaults to 1.0.
        min_pixel_value (float, optional): The minimum allowable pixel value in the image. Defaults to -1.0.

    Returns:
        torch.Tensor: The normalized and clipped image tensor, conforming to the specified p-norm and
        pixel value constraints.
    """

    def __init__(
        self,
        norm_value: float = 30.0,
        p: int = 1,
        max_pixel_value: float = 1.0,
        min_pixel_value: float = -1.0,
    ):
        self.max_value = norm_value
        self.max_pixel_value = float(max_pixel_value)
        self.min_pixel_value = float(min_pixel_value)
        self.p = p
        self.p_norm_constraint = PNormConstraint(norm_value, p)

    def __call__(self, img: torch.Tensor, iteration=None):
        normalized_img = self.p_norm_constraint(img, iteration)
        max_pixel_value = torch.tensor(self.max_pixel_value, dtype=normalized_img.dtype, device=normalized_img.device)
        min_pixel_value = torch.tensor(self.min_pixel_value, dtype=normalized_img.dtype, device=normalized_img.device)
        normalized_img = torch.where(normalized_img > max_pixel_value, max_pixel_value, normalized_img)
        normalized_img = torch.where(normalized_img < min_pixel_value, min_pixel_value, normalized_img)

        return normalized_img
