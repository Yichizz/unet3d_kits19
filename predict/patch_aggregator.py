"""!@file patch_aggregator.py
@brief Patch aggregator functions for aggregating patch predictions and ground truth.
@details This file contains functions for aggregating patch predictions and ground truth to obtain the final segmentation mask.
"""
import torch
import numpy as np


def find_image_size(starts, patch_size):
    """!@brief Find the size of the image given the starting coordinates of patches and the patch size.
    @param starts List of starting coordinates of patches.
    @param patch_size Size of the patch.
    @return image_size Size of the image.
    """
    # find the size of the image
    max_x = max([start[0] for start in starts])
    max_y = max([start[1] for start in starts])
    max_z = max([start[2] for start in starts])
    return (max_x + patch_size[0], max_y + patch_size[1], max_z + patch_size[2])


def aggregate_ground_truth(starts, patch_size, ground_truths):
    """!@brief Aggregate the ground truth segmentation masks from patches.
    @param starts List of starting coordinates of patches.
    @param patch_size Size of the patch.
    @param ground_truths Ground truth segmentation masks.
    @return aggregated_ground_truth Aggregated ground truth segmentation mask.
    """
    if isinstance(starts[0], list) == False:
        starts = [starts]
    image_size = find_image_size(starts, patch_size)
    # create a numpy array to store the aggregated ground truth
    aggregated_ground_truth = torch.zeros(image_size).to(ground_truths.device)
    # ground_truth has shape (num_patches, patch_size[0], patch_size[1], patch_size[2])
    for i in range(len(starts)):
        start = starts[i]
        ground_truth = ground_truths[i]
        aggregated_ground_truth[
            start[0] : start[0] + patch_size[0],
            start[1] : start[1] + patch_size[1],
            start[2] : start[2] + patch_size[2],
        ] = ground_truth
    return (
        aggregated_ground_truth  # shape (image_size[0], image_size[1], image_size[2])
    )


def aggregate_predictions(starts, patch_size, predictions, logits=True):
    """!@brief Aggregate the predicted segmentation masks from patches.
    @param starts List of starting coordinates of patches.
    @param patch_size Size of the patch.
    @param predictions Predicted segmentation masks.
    @param logits Whether the predictions are logits or probabilities.

    @return aggregated_predictions Aggregated predicted segmentation mask.
    """
    if isinstance(starts[0], list) == False:
        starts = [starts]
    image_size = find_image_size(starts, patch_size)
    # create a numpy array to store the aggregated predictions
    num_classes = predictions.shape[
        1
    ]  # (num_patches, num_classes, patch_size[0], patch_size[1], patch_size[2])
    aggregated_predictions = torch.zeros(
        (num_classes, image_size[0], image_size[1], image_size[2])
    ).to(predictions.device)
    # for overlapping patches, we will average the predicted probabilities
    # we first conver logits to probabilities
    if logits:
        predictions = torch.softmax(predictions, dim=1)
    for i in range(len(starts)):
        start = starts[i]
        prediction = predictions[i]
        aggregated_predictions[
            :,
            start[0] : start[0] + patch_size[0],
            start[1] : start[1] + patch_size[1],
            start[2] : start[2] + patch_size[2],
        ] += prediction
    # for non-overlapping patches, the probability still adds up to 1, but we will average the probabilities
    # we need to divide the probabilities by the sum of probabilities at each pixel
    # we will add a small number to the sum to avoid division by zero
    sum_probabilities = (
        aggregated_predictions.sum(dim=0) + 1e-8
    )  # shape (image_size[0], image_size[1], image_size[2])
    aggregated_predictions = aggregated_predictions / sum_probabilities.unsqueeze(0)
    # we now inverse the softmax to logits
    if logits:
        aggregated_predictions = torch.log(aggregated_predictions + 1e-8)
    return aggregated_predictions  # shape (num_classes, image_size[0], image_size[1], image_size[2])
