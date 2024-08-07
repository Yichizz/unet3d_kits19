"""!@file ensemble_predictor.py
@brief Ensemble predictor class for predicting segmentation masks using an ensemble of models.

@details This file contains the EnsemblePredictor class, which is used to predict segmentation masks using an ensemble of models.
"""
import torch
from torch.nn.functional import softmax
from data_preparation.patch_generator import generate_patches
from predict.patch_aggregator import aggregate_predictions


class EnsemblePredictor:
    """!@brief Ensemble predictor class for predicting segmentation masks using an ensemble of models.
        @param models List of models to be used for prediction.
        @param device Device to be used for prediction.
        @param image_array Image array needs to be predicted.

        @return prediction Segmentation mask predicted by the ensemble of models.
    """
    def __init__(self, models, device):
        self.models = models
        self.device = device

    def predict(self, image_array):
        image_patches, starts = generate_patches(
            image_array,
            mask_array=None,
            crop_zero=True,
            patch_size=(80, 160, 160),
            stride=(60, 120, 120),
        )
        starts = [list(starts[i]) for i in range(len(starts))]
        image_patches = (
            torch.from_numpy(image_patches)
            .unsqueeze(1)
            .to(torch.float32)
            .to(self.device)
        )
        patch_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                patch_prediction = model(image_patches)
                patch_prediction = softmax(patch_prediction, dim=1)
            patch_predictions.append(patch_prediction)
            del patch_prediction  # free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        averaged_patch_predictions = torch.stack(patch_predictions).mean(dim=0)
        case_prediction = aggregate_predictions(
            predictions=averaged_patch_predictions,
            starts=starts,
            logits=False,
            patch_size=(80, 160, 160),
        )
        return case_prediction
