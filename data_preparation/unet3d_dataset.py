"""!@file: unet3d_dataset.py
@description: This script defines the UNet3DDataset class.
@details: This script defines the UNet3DDataset class which inherits from the Dataset class of PyTorch.
            This class is used to load the dataset and generate patches for the 3D U-Net model.
"""

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from data_preparation.load_imaging import count_cases, load_case, preprocess, get_arrays


class UNet3DDataset(Dataset):
    """!@brief: Dataset class for 3D U-Net model.
        @param dir: the directory of the dataset
        @param mode: the mode of the dataset, either 'train' or 'test'

        @return: image array and mask array
    """

    def __init__(self, dir, mode="train"):
        self.dir = dir  # eg. './data/'
        self.mode = mode
        self.case_num, self.segmentation_num = count_cases(self.dir)

    def __getitem__(self, idx):
        if self.mode == "train":
            image, mask = load_case(idx, self.dir)
            image, mask = preprocess(
                image,
                mask,
                new_spacing=[3.22, 1.62, 1.62],
                interpolator=sitk.sitkLinear,
                range=(-79, 304),
                subtract=101,
                divide=76.9,
            )
            image_array, mask_array = get_arrays(image, mask)
            return image_array, mask_array
        elif self.mode == "test":
            idx = idx + self.segmentation_num
            image, _ = load_case(idx, self.dir)
            image, _ = preprocess(
                image,
                None,
                new_spacing=[3.22, 1.62, 1.62],
                interpolator=sitk.sitkLinear,
                range=(-79, 304),
                subtract=101,
                divide=76.9,
            )
            image_array, _ = get_arrays(image, None)
            mask_array = np.zeros_like(image_array)
            return image_array, mask_array
        else:
            raise ValueError("mode should be either train or test")

    def __len__(self):
        # count the number of cases in the dataset
        case_num, segmentation_num = count_cases(self.dir)
        if self.mode == "train":
            return segmentation_num
        elif self.mode == "test":
            return case_num - segmentation_num
        else:
            raise ValueError("mode should be either train or test")
