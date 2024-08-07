"""!@file: patch_generator.py
@description: This script generates patches from the image and mask.
@details: This script provides 3 ways of generating patches: 'sw_overlap', 'sw_no_overlap' and 'random'.
            'sw_overlap' generates patches using sliding window with overlap, 'sw_no_overlap' generates patches using sliding window without overlap,
            and 'random' generates patches randomly from the image and mask.
"""

import numpy as np


def pad_image(img, pad_size=(80, 160, 160)):
    """!@brief: Pad the image to make sure the patches cover the entire image.
        @param img: image array
        @param pad_size: the size of the padded image
        @return: padded image
    """
    # dimensions should agree
    assert len(img.shape) == len(pad_size), "Image and pad dimensions should agree."
    if np.all(np.array(pad_size) <= np.array(img.shape)):
        return img  # no need to pad

    pad = np.maximum(np.array(pad_size) - np.array(img.shape), 0)
    pad = [(0, p) for p in pad]
    return np.pad(img, pad, mode="constant")


def compute_patch_num(image_array, patch_size=(80, 160, 160), stride=(40, 80, 80)):
    """!@brief: Compute the number of possible patches in each dimension.
        @param image_array: image array
        @param patch_size: the size of the patch
        @param stride: the stride of the sliding window
        @return: number of patches in each dimension
    """
    # dimensions should agree
    assert (
        len(image_array.shape) == len(patch_size) == len(stride)
    ), "Image, patch and stride dimensions should agree."

    d = (image_array.shape[0] - patch_size[0]) // stride[0] + 1
    h = (image_array.shape[1] - patch_size[1]) // stride[1] + 1
    w = (image_array.shape[2] - patch_size[2]) // stride[2] + 1

    return d, h, w


def get_patch_from_array(
    image_array, patch_size=(80, 160, 160), stride=(40, 80, 80), method="sw_overlap"
):
    """!@brief: Generate patches from the image array.
        @param image_array: image array
        @param patch_size: the size of the patch
        @param stride: the stride of the sliding window
        @param method: the method to generate patches
        @return: patches and starting points
    """
    image_array = pad_image(image_array, patch_size)
    image_shape = np.array(image_array.shape)
    assert len(stride) == len(patch_size), "Stride and patch dimensions should agree."
    assert all(
        [s <= p for s, p in zip(stride, patch_size)]
    ), "Stride should be less than or equal to patch size."
    d, h, w = compute_patch_num(image_array, patch_size, stride)

    patches = []
    if method == "sw_overlap":
        # pad the image to make sure all the pixels in the original image are being considered
        image_array = pad_image(
            image_array, np.array(patch_size) * np.array([d + 1, h + 1, w + 1])
        )
        starts = []
        for i in range(d + 1):
            for j in range(h + 1):
                for k in range(w + 1):
                    start = np.array([i, j, k]) * np.array(stride)
                    end = start + np.array(patch_size)
                    # normal case
                    if np.all(end <= image_shape):
                        patches.append(
                            image_array[
                                start[0] : end[0], start[1] : end[1], start[2] : end[2]
                            ]
                        )
                        starts.append(start)
                    # edge cases
                    else:
                        # if in any dimension, we are out of the image within 1/2 of the patch size, we should keep the patch
                        if np.any(
                            np.logical_and(
                                0 < (end - image_shape),
                                (end - image_shape) <= np.array(patch_size) * 0.5,
                            )
                        ):
                            patches.append(
                                image_array[
                                    start[0] : end[0],
                                    start[1] : end[1],
                                    start[2] : end[2],
                                ]
                            )
                            starts.append(start)

    elif method == "random":
        num_patches = (
            d * h * w + 2
        )  # add 2 to decrease the probability of missing some important regions
        np.random.seed(42)  # for reproducibility
        starts = np.random.randint(
            0, image_shape - np.array(patch_size) + 1, size=(num_patches, 3)
        )
        for start in starts:
            patches.append(
                image_array[
                    start[0] : start[0] + patch_size[0],
                    start[1] : start[1] + patch_size[1],
                    start[2] : start[2] + patch_size[2],
                ]
            )

    elif method == "sw_no_overlap":
        # compute the number of patches in each dimension using no overlap
        d, h, w = compute_patch_num(image_array, patch_size, patch_size)
        image_array = pad_image(
            image_array, np.array(patch_size) * np.array([d + 1, h + 1, w + 1])
        )
        starts = []

        for i in range(d + 1):
            for j in range(h + 1):
                for k in range(w + 1):
                    start = np.array([i, j, k]) * np.array(patch_size)
                    end = start + np.array(patch_size)
                    if np.all(end <= image_shape):
                        patches.append(
                            image_array[
                                start[0] : end[0], start[1] : end[1], start[2] : end[2]
                            ]
                        )
                        starts.append(start)
                    # edge cases
                    else:
                        # if in any dimension, we are out of the image within 1/2 of the patch size, we should keep the patch
                        if np.any(
                            np.logical_and(
                                0 < (end - image_shape),
                                (end - image_shape) <= np.array(patch_size) * 0.5,
                            )
                        ):
                            patches.append(
                                image_array[
                                    start[0] : end[0],
                                    start[1] : end[1],
                                    start[2] : end[2],
                                ]
                            )
                            starts.append(start)
    else:
        raise ValueError(
            "Method should be either 'sw_overlap', 'sw_no_overlap' or 'random'."
        )
    return np.array(patches), np.array(starts)


def generate_patches(
    image_array,
    mask_array=None,
    crop_zero=True,
    patch_size=(80, 160, 160),
    stride=(40, 80, 80),
    method="sw_overlap",
):
    """!@brief: Generate patches from the image and mask arrays.
        @param image_array: image array
        @param mask_array: mask array
        @param crop_zero: whether to crop the image and mask to remove zero regions
        @param patch_size: the size of the patch
        @param stride: the stride of the sliding window
        @param method: the method to generate patches
        @return: patches and starting points
    """
    if crop_zero:
        # crop the dark black regions in the image for reducing the computation
        min_x, min_y, min_z = np.min(
            np.where(image_array - image_array.min() > 1e-3), axis=1
        )
        max_x, max_y, max_z = np.max(
            np.where(image_array - image_array.min() > 1e-3), axis=1
        )
        # crop the image
        image_array = image_array[min_x:max_x, min_y:max_y, min_z:max_z]
        if mask_array is not None:
            mask_array = mask_array[min_x:max_x, min_y:max_y, min_z:max_z]
            mask_array = mask_array.astype(np.uint8)

    image_patches, image_starts = get_patch_from_array(
        image_array, patch_size=patch_size, stride=stride, method=method
    )
    if mask_array is not None:
        mask_patches, mask_starts = get_patch_from_array(
            mask_array, patch_size=patch_size, stride=stride, method=method
        )
        assert np.all(
            [i == j for i, j in zip(image_starts, mask_starts)]
        ), "Image and mask patches should have the same starting points."
        return image_patches, mask_patches, image_starts
    else:
        return image_patches, image_starts
