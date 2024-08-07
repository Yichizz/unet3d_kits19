"""!@file: data_augmentation.py
@brief: Data augmentation for 3D medical images
@details: This script contains the spatial, color, and noise transformations for 3D images.
            Spatial transformations include rotation and scaling. Color transformations include brightness and contrast adjustments.
            Noise transformations include Gaussian and gamma noise addition. The transformations are applied to all slices of the 3D image.
            The transformations are applied identically to all slices to maintain the spatial consistency of the 3D image
"""

import torch
import numpy as np
from torchvision.transforms import functional as F
from torch.distributions import gamma


class SpatialTransform:
    """!@brief: Spatial transformation for 3D medical images
    @params: angle: maximum rotation angle in degrees
            scale: maximum scaling factor
    @return: transformed 3D image
    """

    def __init__(self):
        # default values are 0 and 1 meaning no rotation and no scaling
        self.angle = 0
        self.scale = 1

    def set_params(self, angle, scale):
        # np.random.seed(np.random.randint(0, 1000))
        random_angle = np.random.uniform(-angle, angle)
        random_scale = np.random.uniform(1 - scale, 1 + scale)
        self.angle = random_angle
        self.scale = random_scale
        return self

    def __call__(self, image):
        # image is a 3d tensor of shape (C, D, H, W)
        # apply the same spatial transform to all slices of the image (D slices)
        assert image.dim() == 4, "image must be a 3d tensor of shape (C, D, H, W)"
        for i in range(image.size(1)):
            image[:, i, :, :] = F.affine(
                image[:, i, :, :],
                angle=self.angle,
                scale=self.scale,
                shear=0,
                translate=[0, 0],
            )
        return image


class ColorTransform:
    """!@brief: Color transformation for 3D medical images
    @params: brightness: maximum brightness change
            contrast: maximum contrast change
    @return: transformed 3D image
    """

    def __init__(self):
        # default values are 1 meaning no change in brightness and contrast
        self.brightness = 1
        self.contrast = 1

    def set_params(self, brightness, contrast):
        # unset the seed to get different random values for brightness and contrast
        # np.random.seed(np.random.randint(0, 1000))
        random_brightness = np.random.uniform(1 - brightness, 1 + brightness)
        random_contrast = np.random.uniform(1 - contrast, 1 + contrast)
        self.brightness = random_brightness
        self.contrast = random_contrast
        return self

    def __call__(self, image):
        assert image.dim() == 4, "image must be a 3d tensor of shape (C, D, H, W)"

        # color transformations are also more sensible to apply identically to all slices
        # normalize the image to [0,1]
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        for i in range(image.size(1)):
            image[:, i, :, :] = F.adjust_brightness(image[:, i, :, :], self.brightness)
            image[:, i, :, :] = F.adjust_contrast(image[:, i, :, :], self.contrast)
        # image range after brightness and contrast adjustment is [0,1]
        # denormalize the image to original range
        image = image * (image_max - image_min) + image_min
        return image


class NoiseTransform:
    """!@brief: Noise transformation for 3D medical images
    @params: weights: weights for gaussian and gamma noise
    @return: transformed 3D image
    """

    def __init__(self):
        self.mean = 0
        self.std = 1
        self.shape = 1
        self.scale = 1
        self.weights = [0, 0]  # weights for gaussian and gamma noise
        # default values are 0 and 0 meaning no noise addition

    def set_params(self, weights):
        # np.random.seed(np.random.randint(0, 1000))
        random_gaussian = np.random.uniform(0, weights[0])
        random_gamma = np.random.uniform(0, weights[1])
        self.weights = [random_gaussian, random_gamma]
        return self

    def __call__(self, image):
        assert image.dim() == 4, "image must be a 3d tensor of shape (C, D, H, W)"
        # we can apply different noise to each slice
        # normalize the image to [0,1]
        # image_min = image.min()
        # image_max = image.max()
        # image = (image - image_min) / (image_max - image_min)
        for i in range(image.size(1)):
            image_size = (image.size(2), image.size(3))
            noise_gaussian = torch.normal(mean=self.mean, std=self.std, size=image_size)
            noise_gamma = gamma.Gamma(self.shape, self.scale).sample(image_size)
            image[:, i, :, :] = (
                image[:, i, :, :]
                + noise_gaussian * self.weights[0]
                + noise_gamma * self.weights[1]
            )
        # denormalize the image to original range
        # image = image * (image_max - image_min) + image_min
        return image
