"""!@brief Read parameters from the parameters.ini file and command line arguments
@file read_parameters.py

This file contains functions for reading parameters from the parameters.ini file and command line arguments.
"""
import argparse
import configparser
import os
import torch


def read_train_parameters():
    """!@brief Read parameters from the parameters.ini file and command line arguments
    @return config: configparser.ConfigParser object
    @return gpu_idx: str, GPU number
    @return train_fold: str, fold number for training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param_file",
        type=str,
        default="parameters.ini",
        help="Path to the parameters file",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU number")
    parser.add_argument(
        "--train_fold", type=str, default="1", help="Fold number for training"
    )
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.param_file)
    gpu_idx = args.gpu
    train_fold = args.train_fold
    return config, gpu_idx, train_fold


def read_inference_parameters():
    """!@brief Read parameters from the parameters.ini file and command line arguments
    @return config: configparser.ConfigParser object
    @return gpu_idx: str, GPU number
    @return image_dir: str, path to the image file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param_file",
        type=str,
        default="parameters.ini",
        help="Path to the parameters file",
    )
    parser.add_argument(
        "--image_file",
        type=str,
        default="imaging.nii.gz",
        help="Path to the image file",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU number")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.param_file)
    image_dir = args.image_file
    # if image is not a nifti file
    if not image_dir.endswith(".nii.gz"):
        raise ValueError("Image file is not a nifti file")
    gpu_idx = args.gpu
    return config, image_dir, gpu_idx


def device_diagonistics(gpu_idx):
    """!@brief Check if GPU is available and set the device
    @param gpu_idx: str, GPU number
    @return device: torch.device object
    """
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        device = torch.device("cuda")
        print("Using GPU:", gpu_idx)

    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device


def params_to_dict(config):
    """!@brief Convert parameters from configparser.ConfigParser object to dictionary
    @param config: configparser.ConfigParser object
    @return params: dict, dictionary of parameters
    """
    params = {}
    for section in config.sections():
        params[section] = {}

    # section: hyperparameters_preprocess
    params["hyperparameters_preprocess"]["crop_zero"] = config.getboolean(
        "hyperparameters_preprocess", "crop_zero", fallback=True
    )
    params["hyperparameters_preprocess"]["stride"] = list(
        map(int, config.get("hyperparameters_preprocess", "stride").split(","))
    )
    params["hyperparameters_preprocess"]["patch_generator"] = config.get(
        "hyperparameters_preprocess", "patch_generator", fallback="sw_overlap"
    )

    # section: data augmentation
    params["data_augmentation"]["num_aug"] = config.getint(
        "data_augmentation", "num_aug", fallback=1
    )
    params["data_augmentation"]["biased_sampling"] = config.getboolean(
        "data_augmentation", "biased_sampling", fallback=True
    )

    # section: hyperparameters_training
    params["hyperparameters_training"]["batch_size"] = config.getint(
        "hyperparameters_training", "batch_size", fallback=128
    )
    params["hyperparameters_training"]["n_epochs"] = config.getint(
        "hyperparameters_training", "n_epochs", fallback=100
    )
    params["hyperparameters_training"]["n_folds"] = config.getint(
        "hyperparameters_training", "n_folds", fallback=5
    )
    params["hyperparameters_training"]["early_stopping"] = config.getint(
        "hyperparameters_training", "early_stopping", fallback=10
    )
    if params["hyperparameters_training"]["early_stopping"] == 0:
        params["hyperparameters_training"]["early_stopping"] = False
    params["hyperparameters_training"]["lr_scheduler"] = config.getint(
        "hyperparameters_training", "lr_scheduler", fallback=2
    )

    # section: name
    params["name"]["model_name"] = config.get(
        "name", "model_name", fallback="3d_unet_original"
    )

    return params
