"""!@mainpage 3D U-Net for Kidney Tumor Segmentation
@section train_sec
Train 3D U-Net models for kidney tumor segmentation
@subsection overview Overview
This file contains the main function for training 3D U-Net models for kidney tumor segmentation.

@subsection brief_sec Brief
This script trains 3D U-Net models for kidney tumor segmentation using the KiTS19 dataset.
The script reads the parameters from the configuration file, creates the dataset, model, loss function, and trainer.

The script trains the model using K-Fold cross-validation and saves the results to the results directory.
The script also saves the loss and dice curves to the figures directory.

@subsection usage_sec Usage
The script can be run from the command line using the following command:
python train_unet3d.py --param_file parameters.ini --gpu 0 --train_fold 0

@section inference_sec 
Inference for kidney tumor segmentation using 3D U-Net models

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
import time
import pickle
import warnings
from data_preparation.unet3d_dataset import UNet3DDataset
from unet_3d.unet_3d_plain import UNet3D
from unet_3d.unet_3d_residual import UNet3DResidual
from unet_3d.unet_3d_pre_activation import UNet3DPreActivation
from train.losses import Dice_CE_Loss
from train.engine import KFoldTrainer
from utils.create_folders import create_folders, create_sub_folders
from utils.visualize import plot_losses_curves

# from utils.visualize import plot_losses_curves
from utils.read_parameters import (
    read_train_parameters,
    params_to_dict,
    device_diagonistics,
)


def main():
    # ignore warnings
    warnings.filterwarnings("ignore")
    # create folders
    create_folders()

    # read parameters
    config, gpu_idx, train_fold = read_train_parameters()
    device = device_diagonistics(gpu_idx)
    params = params_to_dict(config)
    if train_fold != "all" and int(train_fold) in range(
        params["hyperparameters_training"]["n_folds"]
    ):
        train_fold = int(train_fold)
    elif train_fold != "all" and int(train_fold) not in range(
        params["hyperparameters_training"]["n_folds"]
    ):
        raise ValueError(
            "Fold number is not correct, please choose a number between 0 and n_folds-1"
        )

    # create dataset
    crop_zero = params["hyperparameters_preprocess"]["crop_zero"]
    stride = params["hyperparameters_preprocess"]["stride"]
    patch_generator = params["hyperparameters_preprocess"]["patch_generator"]
    dataset = UNet3DDataset(dir="data/", mode="train")

    # create model
    model_name = params["name"]["model_name"]
    if model_name == "3d_unet_original":
        model = UNet3D(in_channels=1, out_channels=3, base_channels=30)
    elif model_name == "3d_unet_residual":
        model = UNet3DResidual(in_channels=1, out_channels=3, base_channels=24)
    elif model_name == "3d_unet_pre_activation":
        model = UNet3DPreActivation(in_channels=1, out_channels=3, base_channels=24)
    else:
        raise ValueError("Model name is not correct")

    # create loss function
    loss_fn = Dice_CE_Loss().to(device)

    # create trainer
    num_aug = params["data_augmentation"]["num_aug"]
    biased_sampling = params["data_augmentation"]["biased_sampling"]
    n_epochs = params["hyperparameters_training"]["n_epochs"]
    batch_size = params["hyperparameters_training"]["batch_size"]
    n_splits = params["hyperparameters_training"]["n_folds"]
    early_stopping = params["hyperparameters_training"]["early_stopping"]
    lr_scheduler = params["hyperparameters_training"]["lr_scheduler"]
    if early_stopping == False:
        early_stopping = n_epochs  # never stop early
    # create folders for saving models, results, and figures
    model_dir, results_dir, fig_dir = create_sub_folders(
        model_name, patch_generator, num_aug, biased_sampling
    )

    begin = time.time()
    # create trainer
    trainer = KFoldTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        criterion=loss_fn,
        device=device,
        model_path=model_dir,
        results_dir=results_dir,
        n_splits=n_splits,
        num_epochs=n_epochs,
        early_stopping=early_stopping,
        stride=stride,
        crop_zero=crop_zero,
        patch_generator=patch_generator,
        patch_size=(80, 160, 160),
        num_augmentation=num_aug,
        biased_sampling=biased_sampling,
        lr_scheduler=lr_scheduler,
        verbose=True,
    )
    print("Start training model:", model_name)
    results = trainer.train(fold=train_fold)
    end = time.time()
    print("Training completed in:", end - begin / 60, "minutes")
    # save the results dictionary to results_dir as a pickle file
    with open(results_dir + "/fold_" + str(train_fold) + "_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to:", results_dir)
    # plot the results
    for i in range(n_splits):
        if train_fold != "all" and i != train_fold:
            continue
        else:
            fig_name = fig_dir + "/fold_" + str(i) + "_losses.png"
            fig, _ = plot_losses_curves(
                train_loss=results[i]["train_losses"],
                val_loss=results[i]["val_losses"],
                train_metric=results[i]["train_comp_dices"],
                val_metric=results[i]["val_comp_dices"],
            )
            fig.savefig(fig_name)
    print("Loss and dice curves saved to:", fig_dir)


if __name__ == "__main__":
    main()
