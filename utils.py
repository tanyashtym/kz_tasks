import numpy as np
import os
import matplotlib.pyplot as plt

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
