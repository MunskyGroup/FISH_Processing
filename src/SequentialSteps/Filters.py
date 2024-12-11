import numpy as np
from skimage.io import imread
import tifffile
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from scipy import signal
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl
import trackpy as tp
# import bigfish.segmentation as segmentation

mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import multiprocessing
from smb.SMBConnection import SMBConnection
import socket
import pathlib
import yaml
import shutil
from fpdf import FPDF
import gc
import pickle
import pycromanager as pycro
import pandas as pd
import cellpose
from cellpose import models


import torch
import warnings
# import tensorflow as tf

import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot

from typing import Union

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imsave
import seaborn as sns
from skimage import exposure
import io

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch

    number_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    if number_gpus > 1:  # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.random.randint(0, number_gpus, 1)[0])
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import zipfile
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar

font_props = {'size': 16}
import joypy
from matplotlib import cm
from scipy.ndimage import binary_dilation
import sys
import skimage as sk
from skimage import exposure
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from skimage import exposure
from tifffile import imsave
import copy
from scipy.optimize import curve_fit
from abc import abstractmethod
import dask.array as da


# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, IndependentStepClass # TODO: remove this
from src.Parameters import Parameters

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src.GeneralOutput import OutputClass



#%%
class New_Parameters(OutputClass):
    def append(self, new_params):
        Parameters.update_parameters(new_params)

class FiltersOutputClass(OutputClass):
    def __init__(self, image: np.array):
        super().__init__()
        self.ModifyPipelineData = True
        self.list_images = image

class FilteredImages(SequentialStepsClass):
    def main(self, da, sigma_dict, display_plots: bool=False, **kwargs) -> FiltersOutputClass:
        """
        Main function to run the filters.

        Parameters:
        - da: Dask array with shape [p, t, c, y, x]
        - sigma_dict: Dictionary with sigma values per channel {channel_index: sigma_value}
        - display_plots: Boolean to control plotting

        Returns:
        - output: FiltersOutputClass object
        """
        # Step 1: Apply the filters
        corrected_images = self.average_illumination_profile(da,sigma_dict, display_plots)

        # Step 2: Create the output object
        output = FiltersOutputClass(corrected_images)

        return output

    @abstractmethod
    def average_illumination_profile(self, **kwargs) -> da.array:
        """
        Abstract method to be implemented in the child classes.

        Parameters:
        - kwargs: Dictionary with the required parameters

        Returns:
        - corrected_images: Dask array with shape [p, t, c, y, x]
        """
        pass

class exposure_correction(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, FISHChannel, display_plots: bool = False, **kwargs):
        for f in FISHChannel:
            if display_plots:
                plt.imshow(np.max(image[:, :, :, f], axis=0))
                plt.title(f'Pre exposure correction, channel {f}')
                plt.show()
            rna = np.squeeze(image[:, :, :, f])
            rna = exposure.rescale_intensity(rna, out_range=(0, 1))
            rna = exposure.equalize_adapthist(rna)
            image[:, :, :, f] = exposure.rescale_intensity(rna, out_range=(np.min(image[:, :, :, f]), np.max(image[:, :, :, f])))
            if display_plots:
                plt.imshow(np.max(image[:, :, :, f], axis=0))
                plt.title(f'Post exposure correction, channel {f}')
                plt.show()

        output = filter_output(image)
        output.__class__.__name__ = 'exposure_correction'
        return output

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import dask.array as da

class IlluminationCorrection_Template(IndependentStepClass):
    def __init__(self, da, sigma_dict, display_plots=False, save_profiles=False, save_format="npy", save_dir="profiles"):
        """
        Initialize the IlluminationCorrection class.

        Parameters:
        - da: Dask array with shape [P, T, C, Y, X].
        - sigma_dict: Dictionary with sigma values per channel {channel_index: sigma_value}.
        - display_plots: Boolean to control plotting.
        - save_profiles: Boolean to enable saving of computed profiles.
        - save_format: String indicating format to save profiles ('npy', 'tif', or 'png').
        - save_dir: Directory to save profiles.

        Attributes:
        - da: Dask array to be processed.
        - sigma_dict: Dictionary with smoothing parameters per channel.
        - display_plots: Boolean flag to control plotting of profiles and corrections.
        - save_profiles: Boolean flag to enable saving of profiles.
        - save_format: Format for saving profiles ('npy', 'tif', 'png').
        - save_dir: Directory to save profiles.
        - averaged_profiles: Dask array to store computed averaged illumination profiles.
        - smoothed_profiles: Dask array to store computed smoothed illumination profiles.
        """
        # Initialize placeholders for computed profiles
        self.averaged_profiles = None
        self.smoothed_profiles = None


    def main(self, images, sigma_dict, display_plots):
        """
        Main function to process images and correct illumination.

        Parameters:
        - images: Dask array with shape [P, T, C, Z, Y, X].

        Returns:
        - corrected_da: Dask array with corrected images.
        """

        self.da = images
        self.sigma_dict = sigma_dict
        self.display_plots = display_plots
        # Ensure the save directory exists if saving profiles

        num_channels = self.da.shape[2]  # Extract number of channels
        averaged_profiles = []
        smoothed_profiles = []

        # Step 1: Compute Averaged and Smoothed Illumination Profiles
        for channel, sigma in self.sigma_dict.items():
            print(f"Processing channel {channel} with sigma={sigma}...")

            # Compute averaged and smoothed profiles
            avg_profile, smoothed_profile = self.average_illumination_profile(self.da, channel, sigma_smooth=sigma)

            # Append to profile arrays
            averaged_profiles.append(avg_profile)
            smoothed_profiles.append(smoothed_profile)

            # Display smoothed profile
            if self.display_plots:
                self.show_smoothed_profile(smoothed_profile, channel)

        # Convert profile lists to consolidated Dask arrays
        averaged_profiles = da.stack(averaged_profiles, axis=0)  # Shape [C, Y, X]
        smoothed_profiles = da.stack(smoothed_profiles, axis=0)  # Shape [C, Y, X]

        # Step 2: Correct Images Using Smoothed Profiles
        print("Correcting images...")
        corrected_da = self.correct_image(da, smoothed_profiles)

        # Step 3: Return Corrected Images
        return corrected_da

    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """2D Gaussian function."""
        return offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
        )

    def fit_gaussian_2d(self, illumination_profile, sigma_smooth=200):
        """Fit a 2D Gaussian to the illumination profile and apply additional smoothing."""
        y = np.arange(illumination_profile.shape[0])
        x = np.arange(illumination_profile.shape[1])
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = illumination_profile.ravel()

        # Initial guess for parameters
        initial_guess = (
            illumination_profile.shape[1] / 2,
            illumination_profile.shape[0] / 2,
            illumination_profile.shape[1] / 4,
            illumination_profile.shape[0] / 4,
            np.max(illumination_profile),
            np.min(illumination_profile),
        )

        # Fit Gaussian model
        popt, _ = curve_fit(
            lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset: self.gaussian_2d(
                xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset
            ),
            xdata,
            ydata,
            p0=initial_guess,
            maxfev=10000,
        )

        # Create fitted illumination profile
        fitted_profile = self.gaussian_2d(x, y, *popt).reshape(illumination_profile.shape)
        smoothed_fitted_profile = gaussian_filter(fitted_profile, sigma=sigma_smooth)

        return smoothed_fitted_profile

    def average_illumination_profile(self, da, channel, sigma_smooth=200):
        """
        Compute the averaged illumination profile for a single channel across all images.

        Parameters:
        - da: Dask array with shape [P, T, C, Z, Y, X]
        - channel: int, channel index to process
        - sigma_smooth: int, smoothing factor for Gaussian fitting

        Returns:
        - avg_projection: np.ndarray, averaged projection
        - smoothed_profile: np.ndarray, smoothed illumination profile
        """
        avg_projection = None
        num_images = da.shape[0]  # Number of positions

        for pos_idx in range(num_images):
            image = da[pos_idx]  # Select position
            if image.ndim < 5 or channel >= image.shape[2]:
                print(f"Warning: Skipping position {pos_idx} due to incompatible dimensions for channel {channel}")
                continue

            # Compute mean over positions (p) and time (c)
            projection = image.mean(axis=(0, 2))[channel].compute() #TODO: check if this is correct
            projection = projection.astype(np.float64)

            # Accumulate the projection values
            if avg_projection is None:
                avg_projection = projection
            else:
                avg_projection += projection

        if avg_projection is None:
            raise ValueError("No valid images found for the specified channel.")

        # Average the projection
        avg_projection /= num_images

        # Fit and smooth the illumination profile
        smoothed_profile = self.fit_gaussian_2d(avg_projection, sigma_smooth=sigma_smooth)

        # Normalize smoothed profile so that its maximum value is 1
        smoothed_profile /= np.max(smoothed_profile)

        return avg_projection, smoothed_profile

    def correct_image(self, da, smoothed_profiles):
        """
        Apply the calculated illumination correction to each channel and Z-slice independently.

        Parameters:
        - da: Dask array with shape [P, T, C, Z, Y, X]
        - smoothed_profiles: dict, with channel indices as keys and illumination profiles as values

        Returns:
        - corrected_da: Dask array, illumination-corrected image
        """
        epsilon = 1e-6  # Small constant to avoid division by zero

        if da.ndim < 5:
            raise ValueError("Input must have at least 5 dimensions [P, T, C, Z, Y, X]")

        corrected_da = da.copy()

        for p in range(da.shape[0]):  # Loop over positions
            for c in range(da.shape[2]):  # Loop over channels
                if c not in smoothed_profiles:
                    print(f"Warning: No illumination profile for channel {c}. Skipping correction for this channel.")
                    continue

                # Retrieve the smoothed illumination profile for the current channel
                correction_factor = 1.0 / (smoothed_profiles[c] + epsilon)
                correction_factor /= np.median(correction_factor)  # Normalize by median value

                for z in range(da.shape[3]):  # Loop over Z-slices
                    # Extract all time points for this position, channel, and Z-slice
                    channel_slice = da[p, :, c, z, :, :].compute()  # [T, Y, X] for this Z-slice

                    # Apply correction factor
                    corrected_slice = channel_slice * correction_factor[np.newaxis, :, :]

                    # Rescale intensity for each time point in this Z-slice
                    corrected_slice = np.array([
                        exposure.rescale_intensity(slice_, out_range=(slice_.min(), slice_.max()))
                        for slice_ in corrected_slice
                    ])

                    # Assign corrected slices back to the array
                    corrected_da[p, :, c, z, :, :] = da.from_array(corrected_slice)

        return corrected_da

    def show_illumination_profiles(self, illumination_profiles, corrected_profiles, display_plots: bool = False):
        """Display the reconstructed illumination profiles for each channel before and after correction as heatmaps with contour lines."""
        if display_plots:
            num_channels = len(illumination_profiles)  # Number of channels

            for channel in range(num_channels):
                # Compute illumination profiles into NumPy arrays
                original_profile = illumination_profiles[channel].compute()
                corrected_profile = corrected_profiles[channel].compute()

                # Rescale intensity for visualization (optional, based on range of interest)
                original_profile = exposure.rescale_intensity(
                    original_profile, in_range=(np.percentile(original_profile, 1), np.percentile(original_profile, 99))
                )
                corrected_profile = exposure.rescale_intensity(
                    corrected_profile, in_range=(np.percentile(corrected_profile, 1), np.percentile(corrected_profile, 99))
                )

                # Plot original and corrected illumination profiles
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

                sns.heatmap(original_profile, cmap='hot', cbar=True, ax=axes[0])
                axes[0].set_title(f'Original Illumination Profile - Channel {channel}')
                axes[0].axis('off')
                contours = axes[0].contour(original_profile, colors='white', linewidths=0.5, alpha=0.7)
                axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

                sns.heatmap(corrected_profile, cmap='hot', cbar=True, ax=axes[1])
                axes[1].set_title(f'Corrected Illumination Profile - Channel {channel}')
                axes[1].axis('off')
                contours = axes[1].contour(corrected_profile, colors='white', linewidths=0.5, alpha=0.7)
                axes[1].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

                plt.tight_layout()
                plt.show()

    def show_corrected_max_projection(self, display_plots: bool = False):
        """Display max projections of the original and corrected 3D stack for all channels side-by-side."""
        if display_plots:
            num_channels = self.da.shape[2]  # Number of channels in the 5D array

            for channel in range(num_channels):
                # Compute Z max projections lazily and then convert to NumPy arrays
                original_max_projection = self.da[:, :, channel, :, :, :].max(axis=(0, 3)).compute() #TODO: check if this is correct
                corrected_max_projection = self.corrected_da[:, :, channel, :, :, :].max(axis=(0, 3)).compute()

                # Rescale intensity for better visualization
                original_max_projection = exposure.rescale_intensity(
                    original_max_projection, in_range=(np.percentile(original_max_projection, 1), np.percentile(original_max_projection, 99))
                )
                corrected_max_projection = exposure.rescale_intensity(
                    corrected_max_projection, in_range=(np.percentile(corrected_max_projection, 1), np.percentile(corrected_max_projection, 99))
                )

                # Plot original and corrected max projections
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
                axes[0].imshow(original_max_projection, cmap='hot')
                axes[0].set_title(f'Original Max Projection - Channel {channel}')
                axes[0].axis('off')

                axes[1].imshow(corrected_max_projection, cmap='hot')
                axes[1].set_title(f'Corrected Max Projection - Channel {channel}')
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()



class rescale_images(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, id: int, 
             channel_to_stretch: int = None, stretching_percentile:float = 99.9, 
             display_plots: bool = False, **kwargs):
        # reshape image from zyxc to czyx
        image = np.moveaxis(image, -1, 0)
        # rescale image
        print(image.shape)
        image = stack.rescale(image, channel_to_stretch=channel_to_stretch, stretching_percentile=stretching_percentile)

        # reshape image back to zyxc
        image = np.moveaxis(image, 0, -1)

        if display_plots:
            for c in range(image.shape[3]):
                plt.imshow(np.max(image[:, :, :, c], axis=0))
                plt.title(f'channel {c}')
                plt.show()

        output = filter_output(image)
        output.__class__.__name__ = 'rescale_images'
        return output

        

class remove_background(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, FISHChannel: list[int], id: int, spot_z, spot_yx, voxel_size_z, voxel_size_yx,
             filter_type: str = 'gaussian', sigma: float = None, display_plots:bool = False, 
             kernel_shape: str = 'disk', kernel_size = 200, **kwargs):

        rna = np.squeeze(image[:, :, :, FISHChannel[0]])

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'pre-filtered image')
            plt.show()

        if filter_type == 'gaussian':
            if sigma is None:
                voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
                spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))
                sigma = detection.get_object_radius_pixel(
                        voxel_size_nm=voxel_size_nm, 
                        object_radius_nm=spot_size_nm, 
                        ndim=3 if len(rna.shape) == 3 else 2)
            rna = stack.remove_background_gaussian(rna, sigma=sigma)

        elif filter_type == 'log_filter':
            rna = stack.log_filter(rna, sigma=sigma)

        elif filter_type == 'mean':
            rna = stack.remove_background_mean(np.max(rna, axis=0) if len(rna.shape) > 2 else rna, 
                                               kernel_shape=kernel_shape, kernel_size=kernel_size)
        else:
            raise ValueError('Invalid filter type')
        
        image[:, :, :, FISHChannel[0]] = rna

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'filtered image, type: {filter_type}, sigma: {sigma}')
            plt.show()

        output = filter_output(image)
        output.__class__.__name__ = 'remove_background'
        return output

class IlluminationCorrection(IndependentStepClass):
    def __init__(self):
        """
        Initialize the IlluminationCorrection class.

        Parameters:
        - da: Dask array with shape [P, T, C, Y, X].
        - sigma_dict: Dictionary with sigma values per channel {channel_index: sigma_value}.
        - display_plots: Boolean to control plotting.
        - save_profiles: Boolean to enable saving of computed profiles.
        - save_format: String indicating format to save profiles ('npy', 'tif', or 'png').
        - save_dir: Directory to save profiles.

        Attributes:
        - da: Dask array to be processed.
        - sigma_dict: Dictionary with smoothing parameters per channel.
        - display_plots: Boolean flag to control plotting of profiles and corrections.
        - save_profiles: Boolean flag to enable saving of profiles.
        - save_format: Format for saving profiles ('npy', 'tif', 'png').
        - save_dir: Directory to save profiles.
        - averaged_profiles: Dask array to store computed averaged illumination profiles.
        - smoothed_profiles: Dask array to store computed smoothed illumination profiles.
        """
        # Initialize placeholders for computed profiles
        self.averaged_profiles = None
        self.smoothed_profiles = None

    def main(self, images, sigma_dict, display_plots: bool = False, smoothed_profiles: da = None, **kwargs):
        """
        Main function to process images and correct illumination.

        Parameters:
        - images: Dask array with shape [P, T, C, Z, Y, X].

        Returns:
        - corrected_da: Dask array with corrected images.
        """

        self.da = images
        self.sigma_dict = sigma_dict
        self.display_plots = display_plots
        # Ensure the save directory exists if saving profiles

        


        if smoothed_profiles is None:
            num_channels = self.da.shape[2]  # Extract number of channels
            averaged_profiles = []
            smoothed_profiles = []
            # Step 1: Compute Averaged and Smoothed Illumination Profiles
            for channel, sigma in self.sigma_dict.items():
                print(f"Processing channel {channel} with sigma={sigma}...")

                # Compute averaged and smoothed profiles
                avg_profile, smoothed_profile = self.average_illumination_profile(self.da, channel, sigma_smooth=sigma)

                # Append to profile arrays
                averaged_profiles.append(avg_profile)
                smoothed_profiles.append(smoothed_profile)

                # Display smoothed profile

            # Convert profile lists to consolidated Dask arrays
            averaged_profiles = da.stack(averaged_profiles, axis=0)  # Shape [C, Y, X]
            smoothed_profiles = da.stack(smoothed_profiles, axis=0)  # Shape [C, Y, X]
            self.show_illumination_profiles(averaged_profiles, smoothed_profiles)


        # Step 2: Correct Images Using Smoothed Profiles
        print("Correcting images...")
        corrected_da = self.correct_image(self.da, smoothed_profiles)

        New_Parameters({'images': corrected_da, 'smoothed_profiles': smoothed_profiles})

    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """2D Gaussian function."""
        return offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
        )

    def fit_gaussian_2d(self, illumination_profile, sigma_smooth=200):
        """Fit a 2D Gaussian to the illumination profile and apply additional smoothing."""
        y = np.arange(illumination_profile.shape[0])
        x = np.arange(illumination_profile.shape[1])
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = illumination_profile.ravel()

        # Initial guess for parameters
        initial_guess = (
            illumination_profile.shape[1] / 2,
            illumination_profile.shape[0] / 2,
            illumination_profile.shape[1] / 4,
            illumination_profile.shape[0] / 4,
            np.max(illumination_profile),
            np.min(illumination_profile),
        )

        # Fit Gaussian model
        popt, _ = curve_fit(
            lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset: self.gaussian_2d(
                xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset
            ),
            xdata,
            ydata,
            p0=initial_guess,
            maxfev=10000,
        )

        # Create fitted illumination profile
        fitted_profile = self.gaussian_2d(x, y, *popt).reshape(illumination_profile.shape)
        smoothed_fitted_profile = gaussian_filter(fitted_profile, sigma=sigma_smooth)

        return smoothed_fitted_profile

    def average_illumination_profile(self, da, channel, sigma_smooth=200):
        """
        Compute the averaged illumination profile for a single channel across all images.

        Parameters:
        - da: Dask array with shape [P, T, C, Z, Y, X]
        - channel: int, channel index to process
        - sigma_smooth: int, smoothing factor for Gaussian fitting

        Returns:
        - avg_projection: np.ndarray, averaged projection
        - smoothed_profile: np.ndarray, smoothed illumination profile
        """
        avg_projection = None
        num_images = da.shape[0]  # Number of positions

        for pos_idx in range(num_images):
            image = da[pos_idx]  # Select position
            if image.ndim < 5 or channel >= image.shape[2]:
                print(f"Warning: Skipping position {pos_idx} due to incompatible dimensions for channel {channel}")
                continue

            # Compute mean over positions (p) and time (c)
            projection = image.mean(axis=(0, 2))[channel].compute() #TODO: check if this is correct
            projection = projection.astype(np.float64)

            # Accumulate the projection values
            if avg_projection is None:
                avg_projection = projection
            else:
                avg_projection += projection

        if avg_projection is None:
            raise ValueError("No valid images found for the specified channel.")

        # Average the projection
        avg_projection /= num_images

        # Fit and smooth the illumination profile
        smoothed_profile = self.fit_gaussian_2d(avg_projection, sigma_smooth=sigma_smooth)

        # Normalize smoothed profile so that its maximum value is 1
        smoothed_profile /= np.max(smoothed_profile)

        return avg_projection, smoothed_profile

    def correct_image(self, da, smoothed_profiles):
        """
        Apply the calculated illumination correction to each channel and Z-slice independently.

        Parameters:
        - da: Dask array with shape [P, T, C, Z, Y, X]
        - smoothed_profiles: dict, with channel indices as keys and illumination profiles as values

        Returns:
        - corrected_da: Dask array, illumination-corrected image
        """
        epsilon = 1e-6  # Small constant to avoid division by zero

        if da.ndim < 5:
            raise ValueError("Input must have at least 5 dimensions [P, T, C, Z, Y, X]")

        corrected_da = da.copy()

        for p in range(da.shape[0]):  # Loop over positions
            for c in range(da.shape[2]):  # Loop over channels
                # if c not in range(smoothed_profiles.shape[0]):
                #     print(f"Warning: No illumination profile for channel {c}. Skipping correction for this channel.")
                #     continue

                # Retrieve the smoothed illumination profile for the current channel
                correction_factor = 1.0 / (smoothed_profiles[c] + epsilon)
                correction_factor /= np.median(correction_factor)  # Normalize by median value

                for z in range(da.shape[3]):  # Loop over Z-slices
                    # Extract all time points for this position, channel, and Z-slice
                    channel_slice = da[p, :, c, z, :, :].compute()  # [T, Y, X] for this Z-slice

                    # Apply correction factor
                    corrected_slice = channel_slice * correction_factor[np.newaxis, :, :]

                    # Rescale intensity for each time point in this Z-slice
                    corrected_slice = np.array([
                        exposure.rescale_intensity(slice_, out_range=(slice_.min(), slice_.max()))
                        for slice_ in corrected_slice
                    ])

                    # Assign corrected slices back to the array
                    corrected_da[p, :, c, z, :, :] = da.from_array(corrected_slice)

        return corrected_da

    def show_illumination_profiles(self, illumination_profiles, corrected_profiles):
        """Display the reconstructed illumination profiles for each channel before and after correction as heatmaps with contour lines."""
        if self.display_plots:
            num_channels = len(illumination_profiles)  # Number of channels

            for channel in range(num_channels):
                # Compute illumination profiles into NumPy arrays
                original_profile = illumination_profiles[channel]
                corrected_profile = corrected_profiles[channel]

                # Rescale intensity for visualization (optional, based on range of interest)
                original_profile = exposure.rescale_intensity(
                    original_profile, in_range=(np.percentile(original_profile, 1), np.percentile(original_profile, 99))
                )
                corrected_profile = exposure.rescale_intensity(
                    corrected_profile, in_range=(np.percentile(corrected_profile, 1), np.percentile(corrected_profile, 99))
                )

                # Plot original and corrected illumination profiles
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

                sns.heatmap(original_profile, cmap='hot', cbar=True, ax=axes[0])
                axes[0].set_title(f'Original Illumination Profile - Channel {channel}')
                axes[0].axis('off')
                contours = axes[0].contour(original_profile, colors='white', linewidths=0.5, alpha=0.7)
                axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

                sns.heatmap(corrected_profile, cmap='hot', cbar=True, ax=axes[1])
                axes[1].set_title(f'Corrected Illumination Profile - Channel {channel}')
                axes[1].axis('off')
                contours = axes[1].contour(corrected_profile, colors='white', linewidths=0.5, alpha=0.7)
                axes[1].clabel(contours, inline=True, fontsize=8, fmt="%.2f")

                plt.tight_layout()
                plt.show()

    def show_corrected_max_projection(self):
        """Display max projections of the original and corrected 3D stack for all channels side-by-side."""
        if self.display_plots:
            num_channels = self.da.shape[2]  # Number of channels in the 5D array

            for channel in range(num_channels):
                # Compute Z max projections lazily and then convert to NumPy arrays
                original_max_projection = self.da[:, :, channel, :, :, :].max(axis=(0, 3)).compute() #TODO: check if this is correct
                corrected_max_projection = self.corrected_da[:, :, channel, :, :, :].max(axis=(0, 3)).compute()

                # Rescale intensity for better visualization
                original_max_projection = exposure.rescale_intensity(
                    original_max_projection, in_range=(np.percentile(original_max_projection, 1), np.percentile(original_max_projection, 99))
                )
                corrected_max_projection = exposure.rescale_intensity(
                    corrected_max_projection, in_range=(np.percentile(corrected_max_projection, 1), np.percentile(corrected_max_projection, 99))
                )

                # Plot original and corrected max projections
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
                axes[0].imshow(original_max_projection, cmap='hot')
                axes[0].set_title(f'Original Max Projection - Channel {channel}')
                axes[0].axis('off')

                axes[1].imshow(corrected_max_projection, cmap='hot')
                axes[1].set_title(f'Corrected Max Projection - Channel {channel}')
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()



if __name__ == '__main__':
    matplotlib.use('TKAgg')

    ds = pycro.Dataset(r"C:\Users\Jack\Desktop\H128_Tiles_100ms_5mW_Blue_15x15_10z_05step_2")
    kwargs = {'nucChannel': [0], 
              'FISHChannel': [0],
              'user_select_number_of_images_to_run': 5,

              # rescale images
              'channel_to_stretch': 0,
              }
    compiler = SingleStepCompiler(ds, kwargs)
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0))
    plt.title('original image=========')
    plt.show()
    output = compiler.sudo_run_step(rescale_images)
    compiler.list_images = output.list_images
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0))
    plt.title('rescaled image=========')
    plt.show()
    compiler.sudo_run_step(remove_background)
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0)) 
    plt.title('filtered image=========')
    plt.show()