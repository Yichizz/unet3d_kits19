"""!@file: load_imaging.py
@brief: Module contains functions to load and preprocess imaging data from the KiTS19 dataset

@details: The functions in this module are used to load the imaging and segmentation files, preprocess the imaging data, and convert the SimpleITK objects to numpy arrays.
"""

import SimpleITK as sitk
import numpy as np
import os


def count_cases(data_dir):
    """!@brief: Count the number of cases in the data directory
    @param data_dir: the directory of the data
    @return: the number of cases with imaging files and the number of cases with segmentation files
    """
    assert os.path.exists(data_dir), "data directory does not exist"
    image_file = "imaging.nii.gz"
    seg_file = "segmentation.nii.gz"
    case_images, case_segs = 0, 0
    for case in os.listdir(data_dir):
        if os.path.exists(os.path.join(data_dir, case, image_file)):
            case_images += 1
        if os.path.exists(os.path.join(data_dir, case, seg_file)):
            case_segs += 1
    return case_images, case_segs


def load_case(case_id, data_dir):
    """!@brief: Load the imaging and segmentation files for a case
    @param case_id: the case id
    @param data_dir: the directory of the data
    @return: the imaging and segmentation files as SimpleITK objects
    """
    # case_images, _ = count_cases(data_dir)
    # assert case_id >= 0 and case_id <= case_images and isinstance(case_id, int), f'case_id should be an integer between 0 and {case_images-1}'
    assert os.path.exists(data_dir), "data_dir does not exist"
    case_id_str = f"case_00{case_id:03d}"
    case_dir = os.path.join(data_dir, case_id_str)
    image_dir = os.path.join(case_dir, "imaging.nii.gz")
    mask_dir = os.path.join(case_dir, "segmentation.nii.gz")

    image = sitk.ReadImage(image_dir)
    if os.path.exists(mask_dir):
        mask = sitk.ReadImage(mask_dir)
    else:
        mask = None  # if the mask does not exist, return None for test cases
    return image, mask


def print_header(image, mask=None):
    """!@brief: Print the header information of the imaging and segmentation files
    @param image: the imaging file as a SimpleITK object
    @param mask: the segmentation file as a SimpleITK object
    @return: None
    """
    assert isinstance(image, sitk.Image), "image and should be SimpleITK objects"
    if mask is not None:
        assert isinstance(mask, sitk.Image), "mask should be a SimpleITK object"

    print("Image Header:")
    print("Image Size:", image.GetSize())
    print("Image Spacing:", image.GetSpacing())
    print("Image Origin:", image.GetOrigin())
    print("Image Direction:", image.GetDirection())
    print("Image Pixel Type:", image.GetPixelIDTypeAsString())
    print("Image Pixel Type Value:", image.GetPixelIDValue())

    if mask is not None:
        print("\nMask Header:")
        print("Mask Size:", mask.GetSize())
        print("Mask Spacing:", mask.GetSpacing())
        print("Mask Origin:", mask.GetOrigin())
        print("Mask Direction:", mask.GetDirection())
        print("Mask Pixel Type:", mask.GetPixelIDTypeAsString())
        print("Mask Pixel Type Value:", mask.GetPixelIDValue())
    return None


def resample(
    image,
    mask=None,
    new_spacing=[3.22, 1.62, 1.62],
    new_size=None,
    interpolator=sitk.sitkLinear,
):
    """!@brief: Resample the imaging and segmentation files to a new spacing
    @param image: the imaging file as a SimpleITK object
    @param mask: the segmentation file as a SimpleITK object
    @param new_spacing: the new spacing for resampling
    @param interpolator: the interpolator for resampling
    @return: the resampled imaging and segmentation files as SimpleITK objects
    """
    # assert isinstance(image, sitk.Image),  "image and mask should be SimpleITK objects"
    if mask is not None:
        assert isinstance(mask, sitk.Image), "mask should be a SimpleITK object"
    assert len(new_spacing) == 3, "new_spacing should be a list of length 3"
    assert interpolator in [
        sitk.sitkLinear,
        sitk.sitkNearestNeighbor,
        sitk.sitkBSpline,
    ], "interpolator should be sitkLinear, sitkNearestNeighbor, or sitkBSpline"

    # calculate the new size
    if new_size is None:
        new_size = np.round(
            np.array(image.GetSize()) * np.array(image.GetSpacing()) / new_spacing
        )
        new_size = [int(s) for s in new_size]

    # create the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampled_image = resampler.Execute(image)
    if mask is None:
        return resampled_image, None
    else:
        resampled_seg = resampler.Execute(mask)
        return resampled_image, resampled_seg


def clip_and_normalize(image, range=(-79, 304), subtract=101, divide=76.9):
    """!@brief: Clip and normalize the imaging file
    @param image: the imaging file as a SimpleITK object
    @param range: the range for clipping the image
    @param subtract: the value to subtract from the image
    @param divide: the value to divide the image
    @return: the clipped and normalized imaging file as a SimpleITK object
    """
    assert isinstance(image, sitk.Image), "image should be a SimpleITK object"
    assert len(range) == 2, "range should be a list of length 2"

    # clip the image
    image = sitk.Clamp(image, lowerBound=range[0], upperBound=range[1])
    # normalize the image
    image = (image - subtract) / divide
    return image


def preprocess(
    image,
    mask=None,
    new_spacing=[3.22, 1.62, 1.62],
    interpolator=sitk.sitkLinear,
    range=(-79, 304),
    subtract=101,
    divide=76.9,
):
    """!@brief: Preprocess the imaging and segmentation files
    @param image: the imaging file as a SimpleITK object
    @param mask: the segmentation file as a SimpleITK object
    @param new_spacing: the new spacing for resampling
    @param interpolator: the interpolator for resampling
    @param range: the range for clipping the image
    @param subtract: the value to subtract from the image
    @param divide: the value to divide the image
    @return: the preprocessed imaging file as a SimpleITK object
    """
    assert isinstance(image, sitk.Image), "image and mask should be SimpleITK objects"
    if mask is not None:
        assert isinstance(mask, sitk.Image), "mask should be a SimpleITK object"

    # resample the image and segmentation to 3.22 * 1.62 * 1.62 mm
    resampled_image, resampled_seg = resample(
        image, mask, new_spacing, None, interpolator
    )
    # clip and normalize the image
    preprocessed_image = clip_and_normalize(resampled_image, range, subtract, divide)
    return preprocessed_image, resampled_seg


def get_arrays(image, mask=None):
    """!@brief: Convert SimpleITK objects to numpy arrays
    @param image: the imaging file as a SimpleITK object
    @param mask: the segmentation file as a SimpleITK object
    @return: the imaging and segmentation files as numpy arrays
    """
    assert isinstance(image, sitk.Image), "image and mask should be SimpleITK objects"
    if mask is not None:
        assert isinstance(mask, sitk.Image), "mask should be a SimpleITK object"

    # the direction of the image and segmentation is (-0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    # this means we need to permute the axes of the image and segmentation to display them correctly
    image = sitk.PermuteAxes(image, [2, 1, 0])
    # now the direction of the image and segmentation is (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    # we also need to reverse the direction of the image and segmentation
    image = sitk.Flip(image, [False, False, True])
    image_array = sitk.GetArrayFromImage(image)
    if mask is not None:
        mask = sitk.PermuteAxes(mask, [2, 1, 0])
        mask = sitk.Flip(mask, [False, False, True])
        mask_array = sitk.GetArrayFromImage(mask)
    else:
        mask_array = None
    return image_array, mask_array
