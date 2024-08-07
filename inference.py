"""!@section Inference for kidney tumor segmentation using 3D U-Net models

@subsection overview Overview
This file contains the main function for inference for kidney tumor segmentation using 3D U-Net models.

@subsection brief_sec Brief
This script performs inference for kidney tumor segmentation using 3D U-Net models.
The script reads the parameters from the configuration file, loads the models, and generates predictions for the input image.
The model configurations are read from the parameters.ini file. If the model directory does not exist, the script raises an error.
The predictions are saved as a NIfTI file in the current directory.

@subsection usage_sec Usage
The script can be run from the command line using the following command:
python inference.py --param_file parameters.ini --image_file imaging.nii.gz --gpu 0

@author: Yichi Zhang
@date: 2024.06.28
"""

import os
import numpy as np
import torch
import warnings
import SimpleITK as sitk
from data_preparation.load_imaging import preprocess, get_arrays, resample
from data_preparation.patch_generator import pad_image
from unet_3d.unet_3d_plain import UNet3D
from unet_3d.unet_3d_residual import UNet3DResidual
from unet_3d.unet_3d_pre_activation import UNet3DPreActivation
from utils.read_parameters import (
    read_inference_parameters,
    params_to_dict,
    device_diagonistics,
)
from predict.ensemble_predictor import EnsemblePredictor


def create_models(params, device):
    """!@brief Create 3D U-Net models
    @param params: dict, parameters dictionary
    @param device: torch.device object
    @return models: list, list of 3D U-Net models from K-Fold cross-validation
    """
    # create models
    model_name = params["name"]["model_name"]
    patch_generator = params["hyperparameters_preprocess"]["patch_generator"]
    num_aug = params["data_augmentation"]["num_aug"]
    biased_sampling = params["data_augmentation"]["biased_sampling"]
    model_dir = (
        "models/"
        + model_name
        + "_patch_generator_"
        + patch_generator
        + "_num_aug_"
        + str(num_aug)
        + "_biased_sampling_"
        + str(biased_sampling)
    )
    if not os.path.exists(model_dir):
        raise ValueError("Model directory does not exist, please train the model first")
    else:
        n_fold = len(os.listdir(model_dir))
        if model_name == "3d_unet_original":
            models = [
                UNet3D(in_channels=1, out_channels=3, base_channels=30).to(device)
                for _ in range(n_fold)
            ]
        elif model_name == "3d_unet_residual":
            models = [
                UNet3DResidual(in_channels=1, out_channels=3, base_channels=24).to(
                    device
                )
                for _ in range(n_fold)
            ]
        elif model_name == "3d_unet_pre_activation":
            models = [
                UNet3DPreActivation(in_channels=1, out_channels=3, base_channels=24).to(
                    device
                )
                for _ in range(n_fold)
            ]
        else:
            raise ValueError("Model name not recognized")
    for i in range(n_fold):
        model_path = os.path.join(model_dir, "fold_" + str(i + 1) + ".pth")
        models[i].load_state_dict(torch.load(model_path))
    print("Models loaded successfully")
    return models


def pad_to_orignal_size(case_prediction, original_image_array):
    """!@brief Pad the prediction to the original image size
    @param case_prediction: numpy array, prediction for the case
    @param original_image_array: numpy array, original image
    @return prediction_array: numpy array, prediction padded to the original image size
    """
    image_shape = original_image_array.shape
    prediction_array = np.zeros(image_shape)
    min_x, min_y, min_z = np.min(
        np.where(original_image_array - original_image_array.min() > 1e-3), axis=1
    )
    max_x, max_y, max_z = np.max(
        np.where(original_image_array - original_image_array.min() > 1e-3), axis=1
    )
    # pad case prediction to [MAX-MIN] if in any dimension, we don't have the full prediction
    case_prediction = pad_image(
        case_prediction, (max_x - min_x, max_y - min_y, max_z - min_z)
    )
    # top left corner of the prediction array should be in minx to maxx, miny to maxy, minz to maxz
    prediction_array[min_x:max_x, min_y:max_y, min_z:max_z] = case_prediction[
        : max_x - min_x, : max_y - min_y, : max_z - min_z
    ]
    return prediction_array


def generate_predictions(image_file, device):
    """!@brief Generate predictions for the input image
    @param params: dict, parameters dictionary
    @param image_file: str, path to the input image file
    @param device: torch.device object
    @return case_prediction: SimpleITK object, prediction for the input image
    """
    image_file = sitk.ReadImage(image_file)
    image_preprocessed, _ = preprocess(image_file)
    image_array, _ = get_arrays(image_preprocessed)
    ensemble_predictor = EnsemblePredictor(models, device)
    pred_probs = ensemble_predictor.predict(image_array)
    # convert to 0, 1, 2 labels
    pred_mask = torch.argmax(pred_probs, dim=0).to(torch.int8)
    predictions_array = pred_mask.cpu().numpy()
    predictions = pad_to_orignal_size(predictions_array, image_array).astype(np.uint8)
    assert (
        predictions.shape == image_array.shape
    ), "Predictions and original image should have the same shape"
    # convert to SimpleITK object
    predictions = sitk.GetImageFromArray(
        predictions.transpose(2, 1, 0)
    )  # The voxel spacing of the predictions should be now the same as the input image, which is [3.22, 1.62, 1.62]
    predictions.SetSpacing(image_preprocessed.GetSpacing())
    # set image center as the origin
    predictions.SetOrigin(image_preprocessed.GetOrigin())
    predictions.SetDirection(image_preprocessed.GetDirection())
    predictions = sitk.Flip(predictions, [True, False, False])
    case_prediction = resample(
        image=predictions,
        mask=None,
        new_spacing=image_file.GetSpacing(),
        new_size=image_file.GetSize(),
        interpolator=sitk.sitkNearestNeighbor,
    )[0]
    return case_prediction


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config, image_file, gpu_idx = read_inference_parameters()
    params = params_to_dict(config)
    device = device_diagonistics(gpu_idx)
    models = create_models(params, device)

    predictions = generate_predictions(image_file, device)
    sitk.WriteImage(predictions, "predictions.nii.gz")
    print("Predictions saved successfully")
