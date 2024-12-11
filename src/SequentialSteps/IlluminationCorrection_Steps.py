import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import dask.array as da
from abc import abstractmethod
import os
import sys

# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import  IndependentStepClass  # TODO: remove this
from src.Parameters import Parameters
from src.GeneralOutput import OutputClass

class New_Parameters(OutputClass):
    def append(self, new_params):
        Parameters.update_parameters(new_params)

class IlluminationCorrection(IndependentStepClass):
    def __init__(self):
        """
        Initialize the IlluminationCorrection class.
        """
        super().__init__()

    def main(self, images, sigma_dict, display_plots=False, imported_profiles=None, **kwargs):
        """
        Full pipeline to create profiles, correct images, and visualize.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].
        - sigma_dict: Dictionary of sigma values for smoothing per channel.
        - display_plots: Boolean to control visualization.
        - imported_profiles: ndarray of shape [C, Y, X], precomputed illumination profiles.

        Returns:
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        - illumination_profiles: ndarray of shape [C, Y, X].
        """
        print("Starting illumination correction pipeline...")

        if imported_profiles is not None:
            if not isinstance(imported_profiles, np.ndarray):
                raise TypeError("Imported profiles must be a NumPy array.")
            illumination_profiles = imported_profiles
            print("Using imported illumination profiles.")
        else:
            print("Creating new illumination profiles...")
            illumination_profiles = self.create_illumination_profiles(images, sigma_dict)
            print("New illumination profiles created.")

        print("Applying illumination correction to images...")
        corrected_images = self.apply_correction(images, illumination_profiles)
        print("Illumination correction applied.")

        if display_plots:
            print("Visualizing illumination profiles...")
            self.visualize_profiles(illumination_profiles, corrected_images, sigma_dict)

        print("Illumination correction pipeline complete.")
        New_Parameters({'images': corrected_images, 'illumination_profiles': illumination_profiles})
        return corrected_images, illumination_profiles

    def validate_sigma_dict(self, num_channels, sigma_dict):
        """
        Ensure that sigma_dict has the same length as the number of channels.
        """
        print(f"Validating sigma_dict with {num_channels} channels...")
        if len(sigma_dict) != num_channels:
            raise ValueError(f"Expected sigma_dict to have {num_channels} entries, but got {len(sigma_dict)}.")
        print("sigma_dict validated.")

    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """2D Gaussian function."""
        return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

    def fit_gaussian_2d(self, illumination_profile):
        """Fit a 2D Gaussian to the illumination profile."""
        print("Fitting Gaussian to illumination profile...")
        y = np.arange(illumination_profile.shape[0])
        x = np.arange(illumination_profile.shape[1])
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = illumination_profile.ravel()

        initial_guess = (illumination_profile.shape[1] / 2, illumination_profile.shape[0] / 2,
                         illumination_profile.shape[1] / 4, illumination_profile.shape[0] / 4,
                         np.max(illumination_profile), np.min(illumination_profile))

        popt, _ = curve_fit(lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset:
                            self.gaussian_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset),
                            xdata, ydata, p0=initial_guess)

        fitted_profile = self.gaussian_2d(x, y, *popt).reshape(illumination_profile.shape)
        print("Gaussian fitting complete.")
        return fitted_profile

    def create_illumination_profiles(self, images, sigma_dict):
        """
        Create illumination profiles for each channel.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].

        Returns:
        - illumination_profiles: ndarray of shape [C, Y, X].
        """
        print("Computing max projection along Z-axis...")
        max_projected = images.max(axis=3, keepdims=True)  # Max project along Z shape [P, T, C, 1, Y, X]
        print("Computing median projection across P...")
        median_profile = da.median(max_projected, axis=0).compute()  # Median across P shape [T, C, 1, Y, X]

        num_channels = median_profile.shape[1]
        self.validate_sigma_dict(num_channels, sigma_dict)

        print("Smoothing profiles for each channel...")
        smoothed_profiles = np.stack([
            gaussian_filter(median_profile[0, c, 0], sigma=sigma_dict[c])
            for c in range(num_channels)
        ], axis=0)

        print("Smoothing complete. Returning illumination profiles.")
        return smoothed_profiles

    def apply_correction(self, images, illumination_profiles):
        """
        Apply illumination correction to the input images.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].
        - illumination_profiles: ndarray of shape [C, Y, X], smoothed illumination profiles.

        Returns:
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        """
        print("Preparing correction profiles...")
        epsilon = 1e-6
        correction_profiles = 1.0 / (illumination_profiles + epsilon)

        def correct_block(block, correction_profiles):
            corrected_block = np.zeros_like(block)
            for c in range(block.shape[2]):  # Loop over channels
                correction_profile = correction_profiles[c]
                for z in range(block.shape[3]):  # Loop over Z slices
                    slice_ = block[:, :, c, z, :, :]
                    corrected_slice = slice_ * correction_profile[np.newaxis, np.newaxis, :, :]
                    corrected_block[:, :, c, z, :, :] = corrected_slice
            return corrected_block

        print("Applying correction to image blocks...")
        corrected_images = da.map_blocks(
            correct_block,
            images,
            correction_profiles=correction_profiles,
            dtype=images.dtype
        )
        print("Correction applied to all images.")
        return corrected_images

    def visualize_profiles(self, illumination_profiles, corrected_images, sigma_dict):
        """
        Visualize illumination profiles before and after correction.

        Parameters:
        - illumination_profiles: ndarray of shape [C, Y, X].
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        - sigma_dict: Dictionary of sigma values for smoothing.
        """
        print("Creating smoothed profiles for corrected images...")
        corrected_max_projected = corrected_images.max(axis=3, keepdims=True)  # Max project along Z
        corrected_profiles = self.create_illumination_profiles(corrected_max_projected, sigma_dict)

        for c in range(illumination_profiles.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

            # Original smoothed profile
            sns.heatmap(illumination_profiles[c], cmap='hot', cbar=True, ax=axes[0])
            axes[0].set_title(f'Original Smoothed Illumination Profile - Channel {c}')
            axes[0].axis('off')

            # Add contours to the original profile
            contours = axes[0].contour(
                illumination_profiles[c],
                colors='white',
                linewidths=0.5,
                alpha=0.7,
            )
            axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

            # Corrected smoothed profile
            sns.heatmap(corrected_profiles[c], cmap='hot', cbar=True, ax=axes[1])
            axes[1].set_title(f'Corrected Smoothed Illumination Profile - Channel {c}')
            axes[1].axis('off')

            # Add contours to the corrected profile
            contours = axes[1].contour(
                corrected_profiles[c],
                colors='white',
                linewidths=0.5,
                alpha=0.7,
            )
            axes[1].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

            plt.tight_layout()
            plt.show()
        print("Visualization complete.")
