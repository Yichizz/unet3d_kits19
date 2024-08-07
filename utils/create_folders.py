"""!@file create_folders.py
@brief This file contains functions for creating folders for saving models, results, and figures.
"""
import os

def create_folders():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("figs"):
        os.makedirs("figs")
    return None


def create_sub_folders(model_name, patch_generator, num_aug, biased_sampling):
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
    results_dir = (
        "results/"
        + model_name
        + "_patch_generator_"
        + patch_generator
        + "_num_aug_"
        + str(num_aug)
        + "_biased_sampling_"
        + str(biased_sampling)
    )
    fig_dir = (
        "figs/"
        + model_name
        + "_patch_generator_"
        + patch_generator
        + "_num_aug_"
        + str(num_aug)
        + "_biased_sampling_"
        + str(biased_sampling)
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print('subfolders created')
    return model_dir, results_dir, fig_dir
