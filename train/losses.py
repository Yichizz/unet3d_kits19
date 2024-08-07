"""!@file losses.py
@brief Loss functions for training the model

@detail This file contains the following classes:
1. CompositeDice: Computes the dice loss for kidney and tumor classes
2. Dice_CE_Loss: Computes the composite dice loss and cross entropy loss
"""
import torch
from torch import nn, Tensor
from torch.functional import F
import numpy as np
from torch.nn import CrossEntropyLoss
from torchmetrics import Dice


class CompositeDice(nn.Module):
    """!@brief Composite Dice loss function
    @details This class computes the dice loss for kidney and tumor classes, and the composite dice loss as the average of the two dice losses.
    @input pred_labels: predicted labels
    @input true_labels: true labels
    @return dice_kidney: dice loss for kidney class
    """
    def __init__(self):
        super(CompositeDice, self).__init__()
        # Initialize Dice and BCE loss functions
        self.dice_kidney = Dice(ignore_index=0)
        self.dice_tumor = Dice(ignore_index=0)

    def find_kidney_tumor(self, pred_labels, true_labels):
        # find kidney and tumor labels
        kidney_labels = torch.logical_or(true_labels == 1, true_labels == 2).to(
            torch.int8
        )
        tumor_labels = (true_labels == 2).to(torch.int8)
        # find kidney and tumor predictions
        kidney_preds = torch.logical_or(pred_labels == 1, pred_labels == 2).to(
            torch.int8
        )
        tumor_preds = (pred_labels == 2).to(torch.int8)
        return kidney_labels, tumor_labels, kidney_preds, tumor_preds

    def forward(self, pred_labels, true_labels):
        # calculate dice loss
        # if both predictions and true labels are all zeros, return dice loss of 1
        if torch.sum(pred_labels) == 0 and torch.sum(true_labels) == 0:
            dice_tumor, dice_kidney = torch.tensor(1.0), torch.tensor(1.0)
        else:
            kidney_labels, tumor_labels, kidney_preds, tumor_preds = (
                self.find_kidney_tumor(pred_labels, true_labels)
            )
            dice_kidney = self.dice_kidney(kidney_preds, kidney_labels)
            if torch.sum(tumor_labels) == 0 and torch.sum(tumor_preds) == 0:
                dice_tumor = torch.tensor(1.0)
            else:
                dice_tumor = self.dice_tumor(tumor_preds, tumor_labels)
            # calculate composite dice loss as the average of the two dice losses
        composite_dice = (dice_kidney + dice_tumor) / 2
        return dice_kidney, dice_tumor, composite_dice


class Dice_CE_Loss(nn.Module):
    """!@brief Composite Dice loss and CrossEntropy loss
    @details This class computes the composite dice loss and cross entropy loss.
    @param weight: weight for each class
    @param ignore_index: index to ignore
    @return loss: sum of binary cross entropy loss and composite dice loss
    """
    def __init__(self, weight=Tensor([0.5, 1, 1.5]), ignore_index=-100):
        super(Dice_CE_Loss, self).__init__()
        # Initialize CompositeDice loss function
        self.composite_dice = CompositeDice()
        # Initialize CrossEntropy loss function
        self.ce_loss = CrossEntropyLoss(weight, ignore_index)

    def forward(self, pred_logits, true_labels):
        # pred_logits before softmax: (batch_size, num_classes = 3, depth, height, width)
        # true_labels: (batch_size, 1, depth, height, width)
        true_labels = true_labels.long().squeeze(1)
        # ensure the logits are in a reasonable range
        # pred_logits = torch.clamp(pred_logits, -100, 100) # to avoid overflow
        cross_entropy_loss = self.ce_loss(pred_logits, true_labels)
        # calculate composite dice loss
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_labels = torch.argmax(pred_probs, dim=1).to(torch.int8).squeeze(1)
        _, dice_tumor, composite_dice = self.composite_dice(pred_labels, true_labels)
        # sum of binary cross entropy loss and composite dice loss
        loss = cross_entropy_loss + (1 - composite_dice) + (1 - dice_tumor)

        return loss, composite_dice, dice_tumor
