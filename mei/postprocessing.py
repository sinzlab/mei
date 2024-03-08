import torch


class PNormConstraint:
    def __init__(self, norm_value: float, p: int = 1):
        self.p = p
        self.max_value = norm_value

    def __call__(self, img: torch.Tensor, iteration):
        norm_value = torch.pow(torch.sum(torch.pow(torch.abs(img), self.p)), 1 / self.p)
        if norm_value > self.max_value:
            normalized_img = img * (self.max_value / (norm_value + 1e-12))
        else:
            normalized_img = img
        return normalized_img


class PNormConstraintAndClip:
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

    def __call__(self, img: torch.Tensor, iteration=None):
        norm_value = torch.pow(torch.sum(torch.pow(torch.abs(img), self.p)), 1 / self.p)
        if norm_value > self.max_value:
            normalized_img = img * (self.max_value / (norm_value + 1e-12))
        else:
            normalized_img = img
        normalized_img = normalized_img
        normalized_img = torch.where(normalized_img > self.max_pixel_value, self.max_pixel_value, normalized_img)
        normalized_img = torch.where(normalized_img < self.min_pixel_value, self.min_pixel_value, normalized_img)

        return normalized_img
