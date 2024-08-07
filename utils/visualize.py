"""!@file visualize.py
@brief This file contains functions for visualizing the results.
"""
import numpy as np
import subprocess

try:
    import matplotlib.pyplot as plt
except ImportError:
    # install matplotlib
    subprocess.check_call(["python", "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt


# visualize the median slice for height, width, and depth
# we plot segmentation on top of the image
def plot_medium_slices(image_array, seg_array):
    """!@brief Visualize the median slice for height, width, and depth
    @param image_array: numpy array, 3D image
    @param seg_array: numpy array, 3D segmentation
    @return fig: matplotlib.figure.Figure object
    @return ax: matplotlib.axes.Axes object
    """
    dim = image_array.ndim
    vmin, vmax = image_array.min(), image_array.max()
    assert dim == 3, "Only 3D data supported"
    fig, ax = plt.subplots(1, dim, figsize=(15, 5))
    labels = ["Transverse", "Coronal", "Sagittal"]

    for i in range(dim):
        slice_index = image_array.shape[i] // 2
        slice = np.take(image_array, slice_index, axis=i)
        seg_slice = np.take(seg_array, slice_index, axis=i)
        unique_labels = np.unique(seg_slice)

        ax[i].imshow(slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax[i].contour(
            seg_slice,
            levels=list(unique_labels),
            colors=["blue", "red"],
            linewidths=1,
            linestyles="solid",
        )
        # add legend 0: background, 1: kidney, 2: tumor
        ax[i].legend(
            handles=[
                plt.Line2D([0], [0], color="blue", label="Kidney"),
                plt.Line2D([0], [0], color="red", label="Tumor"),
            ],
            loc="upper right",
        )
        ax[i].axis("off")
        ax[i].set_title(f"Medium {labels[i]} plane")

    return fig, ax


def plot_losses_curves(train_loss, val_loss, train_metric, val_metric):
    """!@brief Plot the loss and metric curves
    @param train_loss: list, training loss
    @param val_loss: list, validation loss
    @param train_metric: list, training metric
    @param val_metric: list, validation metric
    @return fig: matplotlib.figure.Figure object
    @return ax: matplotlib.axes.Axes object
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_loss, label="train loss")
    ax[0].plot(val_loss, label="val loss")
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].tick_params(axis="both", labelsize=12)
    ax[0].legend(fontsize=15)

    ax[1].plot(train_metric, label="train dice")
    ax[1].plot(val_metric, label="val dice")
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Comp Dice", fontsize=12)
    ax[1].tick_params(axis="both", labelsize=12)
    ax[1].legend(fontsize=15)

    return fig, ax
