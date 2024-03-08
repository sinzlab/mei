import numpy as np
from scipy.special import expit


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


# generating random sampling background
def bg_gen(sd=1, decay_power=1, norm_method=0):
    """
    Generates a background image using random frequencies and normalization methods.

    This function creates a background image based on a random spectrum that is scaled according to a power law decay.
    It allows for different normalization methods to adjust the pixel values of the resulting image.

    Parameters:
        sd (float, optional): Standard deviation for the normal distribution used to generate random frequencies.
                               Defaults to 1.
        decay_power (float, optional): Power for the power law decay scaling of the spectrum. Defaults to 1.
        norm_method (int, optional): Specifies the normalization method to be used on the generated image.
                                     - 0: No normalization.
                                     - 1: Linear normalization to adjust pixel values to the range [0, 1].
                                     - 2: Non-linear normalization using the sigmoid function.
                                     Defaults to 0.

    Returns:
        np.ndarray: A normalized 2D array representing the generated background image.
    """
    h = 72
    w = 128
    freqs = rfft2d_freqs(h, w)
    fh, fw = freqs.shape  # (72, 65)

    spectrum_var = sd * np.random.normal(0, 1, (2, fh, fw))
    spectrum = spectrum_var[0] + spectrum_var[1] * 1j  # real, imag

    spertum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power
    spertum_scale *= np.sqrt(w * h)
    scaled_spectrum = spectrum * spertum_scale
    img = np.fft.irfft(scaled_spectrum)  # (72*128)
    if norm_method == 0:
        norm_img = img
    # make pixel range in [0,1]
    # method1 (relative linearly)
    elif norm_method == 1:
        x = 2 * img - 1
        norm_img = x / max(1, max(abs(x).flatten()))
        norm_img = norm_img / 2 + 0.5  # constrain_L_inf(2*t-1)/2 + 0.5

    # method2 (sigmoid, non-linear)
    elif norm_method == 2:
        norm_img = expit(img)

    else:
        raise ValueError("Unknown normalization method, has to be either 0, 1 or 2")

    return norm_img


def bg_wn(mean, std, shape=(72, 128)):
    bg_img = np.random.normal(mean, std, shape)
    rang = max(max(bg_img.flatten()), abs(min(bg_img.flatten())))
    bg_img = bg_img / rang * 2.12  # such that each pixel range in (-2.12,2.12)
    return bg_img


def bg_nat_img(dataloaders, dataset_name):
    images = []
    for tier in ["train", "test", "validation"]:
        for x, y in dataloaders[tier][dataset_name]:
            images.append(x.squeeze().cpu().data.numpy())
    images = np.vstack(images)
    return images
