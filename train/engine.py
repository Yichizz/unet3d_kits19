"""!@file engine.py
@brief This file contains the class KFoldTrainer for training a model using K-Fold cross-validation.
@details The class KFoldTrainer is used to train a model using K-Fold cross-validation.
It trains the model on each fold and evaluates it on the validation set of each fold.
The training process is logged and the results are stored in a dictionary.
The class also provides methods for training and evaluating a single fold.
"""
import numpy as np
import time
import torch
import os
from tqdm import tqdm
from train.losses import CompositeDice
from torch.utils.data import Subset
from torch.nn.functional import softmax
from data_preparation.unet3d_dataloader import TrainDataloader, TestDataloader
from sklearn.model_selection import KFold
from predict.patch_aggregator import aggregate_ground_truth, aggregate_predictions
from torch.optim import Adam


class KFoldTrainer:
    """!KFoldTrainer class for training a model using K-Fold cross-validation.
    @details The class KFoldTrainer is used to train a model using K-Fold cross-validation.
    It trains the model on each fold and evaluates it on the validation set of each fold.
    @param model The model to be trained.
    @param dataset The dataset to be used for training and validation.
    @param batch_size The batch size for training.
    @param criterion The loss function to be used for training.
    @param device The device to be used for training.
    @param model_path The path to save the trained model.
    @param results_dir The directory to save the training results.
    @param n_splits The number of splits for K-Fold cross-validation.
    @param num_epochs The number of epochs for training.
    @param early_stopping The number of epochs for early stopping.
    @param stride The stride for extracting patches.
    @param patch_size The size of the patches.
    @param crop_zero Whether to crop zero regions from the patches.
    @param patch_generator The patch generator to be used for training.
    @param num_augmentation The number of augmentations to be used for training.
    @param biased_sampling Whether to use biased sampling for training.
    @param lr_scheduler The learning rate scheduler to be used for training.
    @param verbose Whether to print the training logs.

    @return The results of the training process.
    """
    def __init__(
        self,
        model,
        dataset,
        batch_size,
        criterion,
        device,
        model_path,
        results_dir,
        n_splits=5,
        num_epochs=100,
        early_stopping=10,
        stride=(80, 120, 120),
        patch_size=(80, 160, 160),
        crop_zero=True,
        patch_generator="random",
        num_augmentation=0,
        biased_sampling=False,
        lr_scheduler=2,
        verbose=False,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device
        self.model_path = model_path
        self.results_dir = results_dir
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.stride = stride
        self.patch_size = patch_size
        self.crop_zero = crop_zero
        self.patch_generator = patch_generator
        self.num_augmentation = num_augmentation
        self.biased_sampling = biased_sampling
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose

    def training_epoch(self, train_dataset):
        # reset random seed for each epoch
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_dataloader = TrainDataloader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            crop_zero=self.crop_zero,
            stride=self.stride,
            patch_size=self.patch_size,
            patch_generator=self.patch_generator,
            num_augmentation=self.num_augmentation,
            biased_sampling=self.biased_sampling,
            shuffle=True,
            num_workers=16,
        )
        self.model.train()
        losses, comp_dices, dice_tumors = [], [], []
        for image, mask in tqdm(train_dataloader, leave=False):
            image, mask = image.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            output_logits = self.model(image)
            loss, comp_dice, dice_tumor = self.criterion(output_logits, mask)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            comp_dices.append(comp_dice.item())
            dice_tumors.append(dice_tumor.item())
            del image, mask, output_logits, loss, comp_dice, dice_tumor
            torch.cuda.empty_cache()
        return np.mean(losses), np.mean(comp_dices), np.mean(dice_tumors)

    def validation_epoch(self, val_dataloader, num_cases=10):
        self.model.eval()
        val_losses, val_comp_dices, val_dice_tumors = [], [], []
        with torch.no_grad():
            for i, (image, mask, starts) in enumerate(val_dataloader):
                if i >= num_cases:
                    break
                else:
                    image, mask = image.to(self.device), mask.to(self.device)
                    output_logits = self.model(image)
                    output_probs = softmax(output_logits, dim=1)
                    # for each case in the validation batch
                    for start in starts:
                        num_patches = len(start)
                        image_mask = aggregate_ground_truth(
                            start,
                            patch_size=(80, 160, 160),
                            ground_truths=mask[:num_patches],
                        )
                        image_pred = aggregate_predictions(
                            start,
                            patch_size=(80, 160, 160),
                            predictions=output_probs[:num_patches],
                            logits=False,
                        )
                        # convert the predicted probabilities to predicted labels
                        image_pred = torch.argmax(image_pred, dim=0).to(torch.int8)
                        loss, comp_dice, tumor_dice = self.criterion(
                            output_logits, mask
                        )
                        val_losses.append(loss.item())
                        val_comp_dices.append(comp_dice.item())
                        val_dice_tumors.append(tumor_dice.item())
                    # delete the variables in the current iteration to avoid memory leak(in GPU)
                    del (
                        image,
                        mask,
                        output_logits,
                        output_probs,
                        image_mask,
                        image_pred,
                        loss,
                        comp_dice,
                        tumor_dice,
                    )
                    torch.cuda.empty_cache()
        return np.mean(val_losses), np.mean(val_comp_dices), np.mean(val_dice_tumors)

    def adjust_learning_rate(
        self, epoch, lr, train_losses, train_comp_dice, epoch_change, min_lr=1e-6
    ):
        if epoch > 11 and train_losses[-11] <= min(train_losses[-10:]) and lr > min_lr:
            if epoch > epoch_change + 5:
                epoch_change = epoch
                if train_comp_dice < 0.5:
                    lr *= 0.2
                elif train_comp_dice < 0.6:
                    lr *= 0.5
                elif train_comp_dice < 0.8:
                    lr *= 0.8
                else:
                    lr *= 0.9
        return lr, epoch_change

    def compute_ema(self, now, prev, alpha=0.9):
        if prev == 0:
            return now
        else:
            return alpha * prev + (1 - alpha) * now

    # another way to adjust the learning rate
    def adjust_learning_rate2(
        self, lr, losses_ema, current_epoch, epoch_change, min_lr=1e-6
    ):
        # if the ema has not improved for 20 epochs, reduce the learning rate by a factor of 0.2
        if (
            current_epoch > 31
            and losses_ema[-31] <= min(losses_ema[-30:])
            and lr > min_lr
            and current_epoch > epoch_change + 5
        ):
            if current_epoch > epoch_change + 5:
                epoch_change = current_epoch
                lr *= 0.2
        return lr, epoch_change

    def train_fold(self, train_dataset, val_dataloader, model_idx):
        # val_cases = next(iter(val_dataloader))
        train_losses, val_losses = [], []
        train_losses_ema = []
        train_comp_dices, val_comp_dices = [], []
        train_tumor_dices, val_tumor_dices = [], []
        lr_history = []
        epoch_change = 0
        train_loss_ema = 0

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
            # train the model for one epoch and validate it
            train_loss, train_comp_dice, train_tumor_dice = self.training_epoch(
                train_dataset
            )
            train_loss_ema = self.compute_ema(train_loss, train_loss_ema)
            val_loss, val_comp_dice, val_tumor_dice = self.validation_epoch(
                val_dataloader
            )
            train_losses.append(train_loss)
            train_losses_ema.append(train_loss_ema)
            val_losses.append(val_loss)
            train_comp_dices.append(train_comp_dice)
            val_comp_dices.append(val_comp_dice)
            train_tumor_dices.append(train_tumor_dice)
            val_tumor_dices.append(val_tumor_dice)
            if self.verbose:
                print(
                    f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Composite Dice: {train_comp_dice:.4f}, Validation Composite Dice: {val_comp_dice:.4f}"
                )
            # update the learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
            if self.lr_scheduler == 1:
                updated_lr, epoch_change = self.adjust_learning_rate(
                    epoch,
                    current_lr,
                    train_losses,
                    train_comp_dice,
                    epoch_change,
                )
            else:
                updated_lr, epoch_change = self.adjust_learning_rate2(
                    current_lr, train_losses_ema, epoch, epoch_change
                )
                
            if updated_lr != current_lr:
                self.optimizer.param_groups[0]["lr"] = updated_lr
                print(f"Learning rate reduced to {updated_lr} at epoch {epoch+1}")
            if updated_lr < 1e-6 and val_comp_dice + 0.1 < train_comp_dice:
                print(f"Early stopping at epoch {epoch+1} due to overfitting")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_path, f"fold_{model_idx + 1}.pth"),
                )
                break
            # check for early stopping
            if epoch > self.early_stopping:
                if val_losses[-self.early_stopping] <= min(
                    val_losses[-self.early_stopping :]
                ):
                    print(
                        f"Early stopping at epoch {epoch+1} due to no improvement in validation loss"
                    )
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_path, f"fold_{model_idx + 1}.pth"),
                    )
                    break
            # empty used variables to avoid memory leak
            del (
                train_loss,
                train_comp_dice,
                val_loss,
                val_comp_dice,
                train_tumor_dice,
                val_tumor_dice,
            )
            torch.cuda.empty_cache()
        # save the model after training
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_path, f"fold_{model_idx + 1}.pth"),
        )
        return (
            train_losses,
            val_losses,
            train_comp_dices,
            val_comp_dices,
            train_tumor_dices,
            val_tumor_dices,
            lr_history,
            train_losses_ema,
        )

    def evaluate_fold(self, val_dataloader):
        # evaluate the model on the entire validation set
        self.model.eval()
        val_kidney_dice, val_tumor_dice = 0, 0
        num_cases = 0
        metric = CompositeDice().to(self.device)
        with torch.no_grad():
            for image, mask, starts in tqdm(val_dataloader, leave=True):
                image, mask = image.to(self.device), mask.to(self.device)
                output_logits = self.model(image)
                output_probs = softmax(output_logits, dim=1)
                for start in starts:
                    num_patches = len(start)
                    image_mask = aggregate_ground_truth(
                        start,
                        patch_size=(80, 160, 160),
                        ground_truths=mask[:num_patches],
                    )
                    image_pred = aggregate_predictions(
                        start,
                        patch_size=(80, 160, 160),
                        predictions=output_probs[:num_patches],
                        logits=False,
                    )
                    image_pred = torch.argmax(image_pred, dim=0).to(torch.int8)
                    kidney_dice, tumor_dice, _ = metric(
                        image_pred.unsqueeze(0), image_mask.unsqueeze(0)
                    )
                    val_kidney_dice += kidney_dice
                    val_tumor_dice += tumor_dice
                    num_cases += 1
                del (
                    image,
                    mask,
                    output_logits,
                    output_probs,
                    image_mask,
                    image_pred,
                    kidney_dice,
                    tumor_dice,
                )
                torch.cuda.empty_cache()
            val_kidney_dice /= num_cases
            val_tumor_dice /= num_cases
            val_composite_dice = (val_kidney_dice + val_tumor_dice) / 2
        return val_kidney_dice, val_tumor_dice, val_composite_dice

    def train(self, fold):  # fold = int / 'all'
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # create a dictionary to store the training and validation losses
        fold_results = {}
        initial_weights = self.model.state_dict()
        if fold == "all":
            fold = [i for i in range(self.n_splits)]
        elif type(fold) == int:
            fold = [fold]
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            if fold_idx in fold:
                train_dataset = Subset(self.dataset, train_idx)
                val_dataset = Subset(self.dataset, val_idx)
                val_dataloader = TestDataloader(
                    dataset=val_dataset,
                    batch_size=1,
                    crop_zero=True,
                    stride=(60, 120, 120),
                    patch_size=(80, 160, 160),
                    shuffle=False,
                    num_workers=16,
                )
                self.model = self.model.to(self.device)
                self.model.load_state_dict(
                    initial_weights
                )  # reset the model weights for each fold
                self.optimizer = Adam(
                    self.model.parameters(), lr=5e-4, weight_decay=3e-5
                )
                if self.verbose:
                    print(f"Training fold {fold_idx+1}")
                (
                    train_losses,
                    val_losses,
                    train_comp_dices,
                    val_comp_dices,
                    train_tumor_dices,
                    val_tumor_dices,
                    lr_history,
                    train_losses_ema,
                ) = self.train_fold(train_dataset, val_dataloader, fold_idx)
                if self.verbose:
                    print(
                        f"Training finnished for fold {fold_idx+1} of {self.n_splits}, start evaluation on the whole validation set."
                    )
                val_kidney_dice, val_tumor_dice, val_composite_dice = (
                    self.evaluate_fold(val_dataloader)
                )
                if self.verbose:
                    print(
                        f"Validation Kidney Dice: {val_kidney_dice:.4f}, Validation Tumor Dice: {val_tumor_dice:.4f}, Validation Composite Dice: {val_composite_dice:.4f}"
                    )
                fold_results[fold_idx] = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_comp_dices": train_comp_dices,
                    "val_comp_dices": val_comp_dices,
                    "train_tumor_dices": train_tumor_dices,
                    "val_tumor_dices": val_tumor_dices,
                    "lr_history": lr_history,
                    "train_losses_ema": train_losses_ema,
                    "val_kidney_dice": val_kidney_dice,
                    "val_tumor_dice": val_tumor_dice,
                    "val_composite_dice": val_composite_dice,
                }
                # empty the cache to avoid memory leak
                torch.cuda.empty_cache()
        return fold_results
