"""!@file: unet3d_dataloader.py
@brief: Dataloader for 3D U-Net
@details: This script contains the dataloader for the 3D U-Net model.
            The dataloader is used to load the 3D images and masks, apply data augmentation, and return the batch of images and masks.
            The dataloader uses the SpatialTransform, ColorTransform, and NoiseTransform classes from the data_augmentation.py script.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preparation.patch_generator import generate_patches
from data_preparation.data_augmentation import (
    SpatialTransform,
    ColorTransform,
    NoiseTransform,
)


def collate_fn(batch):
    images, masks = zip(*batch)
    return images, masks


class TrainDataloader:
    """!@brief: Training Dataloader for 3D U-Net
    @params: dataset: dataset object
                batch_size: batch size
                crop_zero: crop dark margins
                patch_generator: patch generation method
                stride: stride of the sliding window
                patch_size: size of the patch
                num_augmentation: number of random augmentations
                biased_sampling: increase the exposure of tumor patches
                shuffle: shuffle the dataset
                num_workers: number of workers
                pin_memory: pin memory
    @return: dataloader object
    """

    def __init__(
        self,
        dataset,
        batch_size,
        crop_zero=True,
        patch_generator="sw_overlap",
        stride=(60, 120, 120),
        patch_size=(80, 160, 160),
        num_augmentation=0,
        biased_sampling=True,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.crop_zero = crop_zero
        self.patch_generator = patch_generator
        self.stride = stride
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.num_augmentation = num_augmentation
        self.biased_sampling = biased_sampling
        assert (
            isinstance(self.num_augmentation, int) and self.num_augmentation <= 3
        ), "num_augmentation must be an integer between 0 and 3"
        self.spatial_transform = SpatialTransform()
        self.color_transform = ColorTransform()
        self.noise_transform = NoiseTransform()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def duplicate_tumor_patches(self, image_patches, mask_patches):
        for i in range(len(mask_patches)):
            if torch.any(mask_patches[i] == 2):
                # add 3 more copies of the tumor patch
                image_patches = torch.cat(
                    [
                        image_patches,
                        image_patches[i].unsqueeze(0),
                        image_patches[i].unsqueeze(0),
                        image_patches[i].unsqueeze(0),
                    ],
                    dim=0,
                )
                mask_patches = torch.cat(
                    [
                        mask_patches,
                        mask_patches[i].unsqueeze(0),
                        mask_patches[i].unsqueeze(0),
                        mask_patches[i].unsqueeze(0),
                    ],
                    dim=0,
                )
        return image_patches, mask_patches

    def patchify(self, image, mask):
        # generate patches
        image_patches, mask_patches, _ = generate_patches(
            image,
            mask,
            crop_zero=self.crop_zero,
            patch_size=self.patch_size,
            stride=self.stride,
            method=self.patch_generator,
        )
        image_patches = torch.from_numpy(image_patches).unsqueeze(1).to(torch.float32)
        mask_patches = torch.from_numpy(mask_patches).unsqueeze(1).to(torch.int8)
        # we want to increase the exposure of tumor patches in the batch
        # so we duplicate the tumor patches
        if self.biased_sampling:
            image_patches, mask_patches = self.duplicate_tumor_patches(
                image_patches, mask_patches
            )
        idx = torch.randperm(image_patches.size(0))
        image_patches = image_patches[idx]
        mask_patches = mask_patches[idx]
        return image_patches, mask_patches

    def check_and_fix(self, image, mask):
        # check if there is any NaN or infinite value in the image after augmentation
        # if so replace it with the 0 value
        if torch.isnan(image).any() or torch.isinf(image).any():
            image[torch.isnan(image)] = 0
            image[torch.isinf(image)] = 0
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            mask[torch.isnan(mask)] = 0
            mask[torch.isinf(mask)] = 0
        return image, mask

    def augment(self, patch, num_augmentation):
        """!@brief: Apply random augmentations to the patch
        @params: patch: tuple of image and mask
                    num_augmentation: number of random augmentations
        @return: augmented image and mask
        """
        # for each patch in the dataloader, apply num_augmentation random augmentations
        assert (
            num_augmentation >= 1
        ), "num_augmentation must be at least 1 if augment is called"
        image, mask = patch

        random_augs = np.random.choice(
            ["spatial", "intensity", "noise"], num_augmentation, replace=False
        )
        # created different copies of augemented patches
        images, masks = [image], [mask]
        if "spatial" in random_augs:
            self.spatial_transform = self.spatial_transform.set_params(
                angle=10, scale=0.1
            )
            image_clone = image.clone()
            mask_clone = mask.clone()
            images.append(self.spatial_transform(image_clone))
            masks.append(self.spatial_transform(mask_clone))
        if "intensity" in random_augs:
            image_clone = image.clone()
            self.color_transform = self.color_transform.set_params(
                brightness=0.5, contrast=0.5
            )
            images.append(self.color_transform(image_clone))
            masks.append(mask)
        if "noise" in random_augs:
            image_clone = image.clone()
            self.noise_transform = self.noise_transform.set_params(weights=[0.3, 0])
            images.append(self.noise_transform(image_clone))
            masks.append(mask)

        return images, masks

    def __iter__(self):
        # return the dataloader iterator
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        # return the next batch of patches
        try:
            images, masks = next(self.iterator)
        except StopIteration:
            raise StopIteration

        image_patches, mask_patches = [], []
        for image, mask in zip(images, masks):
            image_patch, mask_patch = self.patchify(image, mask)
            # random choose 1 patch from the all patches
            idx = torch.randint(0, len(image_patch), (1,))
            image_selected = image_patch[idx]
            mask_selected = mask_patch[idx]
            image_patches.append(image_selected)
            mask_patches.append(mask_selected)
        # we only need 1 random patch from the batch
        image_patches = torch.cat(image_patches, dim=0)
        mask_patches = torch.cat(mask_patches, dim=0)
        batch = (image_patches, mask_patches)

        # apply augmentations
        if self.num_augmentation >= 1:
            aug_imgs, aug_masks = [], []
            for i in range(len(batch[0])):
                aug_img, aug_mask = self.augment(
                    (batch[0][i], batch[1][i]), self.num_augmentation
                )
                aug_imgs.extend(aug_img)
                aug_masks.extend(aug_mask)
            # convert the list of augmented images and masks to a tensor
            aug_imgs = torch.cat(aug_imgs).unsqueeze(1).to(torch.float32)
            aug_masks = torch.cat(aug_masks).unsqueeze(1).to(torch.int8)
            aug_imgs, aug_masks = self.check_and_fix(aug_imgs, aug_masks)
            batch = (aug_imgs, aug_masks)
        return batch

    def __len__(self):
        return len(self.dataloader)


class TestDataloader:
    """!@brief: Test Dataloader for 3D U-Net
    @params: dataset: dataset object
                batch_size: batch size
                crop_zero: crop dark margins
                stride: stride of the sliding window
                patch_size: size of the patch
                shuffle: shuffle the dataset
                num_workers: number of workers
                pin_memory: pin memory
    @return: dataloader object
    """

    def __init__(
        self,
        dataset,
        batch_size,
        crop_zero=True,
        stride=(40, 80, 80),
        patch_size=(80, 160, 160),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.crop_zero = crop_zero
        self.stride = stride
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def patchify(self, image, mask):
        # generate patches
        image_patches, mask_patches, starts = generate_patches(
            image,
            mask,
            crop_zero=self.crop_zero,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        image_patches = torch.from_numpy(image_patches).unsqueeze(1).to(torch.float32)
        mask_patches = torch.from_numpy(mask_patches).unsqueeze(1).to(torch.int8)
        idx = torch.randperm(image_patches.size(0))
        image_patches = image_patches[idx]
        mask_patches = mask_patches[idx]
        starts = [list(starts[i]) for i in idx]
        return image_patches, mask_patches, starts

    def __iter__(self):
        # return the dataloader iterator
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        # return the next batch of patches
        try:
            image, mask = next(self.iterator)
        except StopIteration:
            raise StopIteration
        image_patches, mask_patches, starts = [], [], []
        for image, mask in zip(image, mask):
            image_patch, mask_patch, start = self.patchify(image, mask)
            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            starts.append(start)
        # concatenate the list of patches to a tensor
        image_patches = torch.cat(image_patches, dim=0)
        mask_patches = torch.cat(mask_patches, dim=0)
        batch = (image_patches, mask_patches, starts)
        return batch

    def __len__(self):
        return len(self.dataloader)
