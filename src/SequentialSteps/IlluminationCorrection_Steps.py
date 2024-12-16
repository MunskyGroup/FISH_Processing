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

#%%
from skimage.filters import threshold_otsu
import dask.array as da
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from dask import delayed

class IlluminationCorrection_BGFG(IndependentStepClass):
    def __init__(self):
        """
        Initialize the IlluminationCorrection class.
        """
        super().__init__()

    def otsu_threshold(self, max_projection):
        """
        Apply Otsu's method to separate foreground and background.

        Parameters:
        - max_projection: ndarray of shape [Y, X], 2D max projection of an image.

        Returns:
        - foreground_mask: Binary mask of the foreground.
        - background_mask: Binary mask of the background.
        """
        print("Computing Otsu threshold...")
        threshold_value = threshold_otsu(max_projection)
        print(f"Otsu threshold value: {threshold_value}")

        # Create binary masks
        foreground_mask = max_projection > threshold_value
        background_mask = ~foreground_mask

        return foreground_mask, background_mask

    # def create_illumination_profiles(self, images, sigma_dict):
    #     """
    #     Create illumination profiles for each channel.

    #     Parameters:
    #     - images: Dask array of shape [P, T, C, Z, Y, X].

    #     Returns:
    #     - illumination_profiles: ndarray of shape [C, Y, X].
    #     """
    #     print("Computing max projection along Z-axis for all channels...")
    #     max_projected = images.max(axis=3)  # Lazy operation [P, T, C, Y, X]

    #     print("Computing median across P dimension...")
    #     median_projection = da.median(max_projected, axis=0)  # Lazy operation [T, C, Y, X]

    #     num_channels = median_projection.shape[1]
    #     self.validate_sigma_dict(num_channels, sigma_dict)

    #     print("Smoothing profiles and applying Otsu threshold for each channel...")
    #     illumination_profiles = []
    #     for c in range(num_channels):
    #         print(f"Processing channel {c}...")

    #         # Delayed computation for Otsu and smoothing
    #         channel_projection = delayed(median_projection)[0, c]  # Extract channel projection
    #         foreground_mask = delayed(self.otsu_threshold)(channel_projection)[0]
    #         smoothed_profile = delayed(gaussian_filter)(channel_projection * foreground_mask, sigma=sigma_dict[c])

    #         illumination_profiles.append(smoothed_profile)

    #     # Stack profiles (forcing computation)
    #     illumination_profiles = da.stack([da.from_delayed(profile, shape=(None, None), dtype=np.float32)
    #                                     for profile in illumination_profiles], axis=0)
    #     print("Illumination profiles created.")
    #     return illumination_profiles.compute()  # Trigger computation

    def create_illumination_profiles(self, images, sigma_dict):
        """
        Create illumination profiles for each channel.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].

        Returns:
        - illumination_profiles: ndarray of shape [C, Y, X].
        """
        print("Computing max projection along Z-axis for all channels...")
        # Compute max projection along Z-axis once for efficiency
        max_projected = images.max(axis=3)  # Shape [P, T, C, Y, X]

        print("Computing median across P dimension...")
        median_projection = da.median(max_projected, axis=0)  # Shape [T, C, Y, X]
        median_projection_computed = median_projection.compute()  # Convert to NumPy array

        num_channels = median_projection_computed.shape[1]
        self.validate_sigma_dict(num_channels, sigma_dict)

        print("Smoothing profiles and applying Otsu threshold for each channel...")
        illumination_profiles = []
        for c in range(num_channels):
            print(f"Processing channel {c}...")
            # Extract channel-specific max projection (median value for [T, Y, X])
            channel_projection = median_projection_computed[0, c]

            # Apply Otsu thresholding
            foreground_mask, _ = self.otsu_threshold(channel_projection)

            # Smooth only the foreground for illumination profile
            smoothed_profile = gaussian_filter(channel_projection * foreground_mask, sigma=sigma_dict[c])
            illumination_profiles.append(smoothed_profile)

        illumination_profiles = np.stack(illumination_profiles, axis=0)  # Stack profiles into [C, Y, X]
        print("Illumination profiles created.")
        return illumination_profiles

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

        # Validate sigma_dict
        self.validate_sigma_dict(images.shape[2], sigma_dict)

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

        def correct_block(block, profiles):
            corrected_block = np.zeros_like(block)
            for c in range(block.shape[2]):  # Loop over channels
                correction_profile = profiles[c]
                for z in range(block.shape[3]):  # Loop over Z slices
                    slice_ = block[:, :, c, z, :, :]
                    corrected_slice = slice_ * correction_profile[np.newaxis, np.newaxis, :, :]
                    corrected_block[:, :, c, z, :, :] = corrected_slice
            return corrected_block

        print("Applying correction to image blocks...")
        corrected_images = da.map_blocks(
            correct_block,
            images,
            correction_profiles,
            dtype=images.dtype,
            chunks=images.chunks
        )
        print("Correction applied to all images.")
        return corrected_images

    
    def visualize_profiles(self, illumination_profiles, corrected_images, sigma_dict):
        """
        Visualize illumination profiles before and after correction, including foreground masks.

        Parameters:
        - illumination_profiles: ndarray of shape [C, Y, X].
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        - sigma_dict: Dictionary of sigma values for smoothing.
        """
        print("Creating smoothed profiles for corrected images...")
        corrected_max_projected = corrected_images.max(axis=3, keepdims=True)  # Max project along Z
        corrected_profiles = self.create_illumination_profiles(corrected_max_projected, sigma_dict)

        for c in range(illumination_profiles.shape[0]):
            print(f"Visualizing channel {c}...")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

            # Original smoothed illumination profile
            sns.heatmap(illumination_profiles[c], cmap='hot', cbar=True, ax=axes[0])
            axes[0].set_title(f'Original Illumination Profile - Channel {c}')
            axes[0].axis('off')

            # Foreground mask (Otsu threshold)
            otsu_threshold = threshold_otsu(illumination_profiles[c])
            foreground_mask = illumination_profiles[c] > otsu_threshold
            sns.heatmap(foreground_mask, cmap='gray', cbar=False, ax=axes[1])
            axes[1].set_title(f'Foreground Mask (Otsu) - Channel {c}')
            axes[1].axis('off')

            # Corrected smoothed profile
            sns.heatmap(corrected_profiles[c], cmap='hot', cbar=True, ax=axes[2])
            axes[2].set_title(f'Corrected Illumination Profile - Channel {c}')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

        print("Visualization complete.")

