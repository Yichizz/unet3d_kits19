"""@package data_preparation
@file __init__.py
@brief Initialization file for the module/package.

This package provides functionalities related to data preparation. 
@author: Yichi Zhang
@date: 2024.06.27

@@section intro_sec Introduction
This module/package contains classes and functions for data preparation tasks. 
It includes downloading the KiTS19 dataset, loading images and segmentations, generating
patches, applying data augmentation, and creating data sets and data loaders.


@section usage_sec Usage
For downloading the KiTS19 dataset, use modules in get_imaging.py
For loading images and segmentations and adjusting the intensity or voxel spacing, use modules in load_imaging.py
For generating patches, use modules in patch_generator.py
For applying data augmentation, use augmentaion classes in data_augmentation.py
For creating data sets and data loaders, use classes in unet3d_dataset.py and unet3d_dataloader.py

@note This module/package is developed based on the KiTS19 dataset, and thus the functionalities are tailored to the KiTS19 dataset.
"""

