import numpy as np
from typing import Sequence, Optional
import os
import matplotlib.pyplot as plt
import math

def load_npy_array(file_path) -> np.ndarray:
    """Load a numpy array from the given .npy file."""
    return np.load(file_path)


def normalize_channels_for_visualization(
    image: np.ndarray,
    clip_percent: tuple[float, float] = (2.0, 98.0)
) -> np.ndarray:
    """Normalize each channel by clipping low/high percentiles."""
    lower, upper = clip_percent
    mins = np.percentile(image, lower, axis=(0, 1))
    maxs = np.percentile(image, upper, axis=(0, 1))
    scales = np.where(maxs > mins, maxs - mins, 1)
    normed = (image - mins) / scales
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def prepare_true_color_composite(raw_image: np.ndarray) -> np.ndarray:
    """Scale first three bands and convert BGR to RGB."""
    composite = raw_image[..., :3] * 10000
    return composite[..., ::-1]


def display_side_by_side(
    img: np.ndarray,
    mask: np.ndarray,
    mask_cmap: str = 'gray',
    titles: tuple[str, str] = ('True-Color Composite', 'Mask')
) -> None:
    """
    Display the true-color composite and mask next to each other in one figure.
    """
    # Create a single row with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    # Plot the RGB image
    axes[0].imshow(img)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # Plot the mask
    axes[1].imshow(mask, cmap=mask_cmap)
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    # Render the combined figure
    plt.tight_layout()
    plt.show()

def show_images(
    images: Sequence[np.ndarray],
    cmaps: Optional[Sequence[Optional[str]]] = None,
    titles: Optional[Sequence[str]] = None,
    max_per_row: int = 2,
    figsize: tuple[float, float] = (8, 8)
) -> None:
    """
    Display up to 4 images in a grid that adapts to the number of images.

    Parameters
    ----------
    images : sequence of ndarray
        The images to show.
    cmaps : sequence of str or None, optional
        Colormaps for each image (e.g. 'gray'); use None for RGB.
        If not provided, all images are shown with their default colormap.
    titles : sequence of str, optional
        Titles for each subplot. If not provided, no titles are set.
    max_per_row : int, default=2
        Maximum number of columns in the grid.
    figsize : (width, height), default=(8, 8)
        Base figure size; will be scaled by rows and columns.
    """
    n = len(images)
    if not (1 <= n <= 4):
        raise ValueError("This function supports between 1 and 4 images.")

    # defaults
    cmaps = cmaps or [None] * n
    titles = titles or [""] * n

    # determine grid
    cols = min(max_per_row, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize[0] * cols, figsize[1] * rows))
    # flatten in case of 1×1 or 1D layouts
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # plot each image
    for ax, img, cmap, title in zip(axes, images, cmaps, titles):
        ax.imshow(img, cmap=cmap)
        if title:
            ax.set_title(title)
        ax.axis('off')

    # hide any unused axes
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

