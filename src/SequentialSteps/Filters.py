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

from src import SequentialStepsClass, SingleStepCompiler, IndependentStepClass # TODO: remove this

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src.GeneralOutput import OutputClass



#%%


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

# TODO: remove this?
# class illumination_correction_output(StepOutputsClass):
#     def __init__(self, images: list):
#         super().__init__()
#         self.ModifyPipelineData = True
#         # Store the images directly as a list of corrected images
#         self.corrected_images = images

#     def append(self, new_output):
#         if new_output and isinstance(new_output, illumination_correction_output):
#             self.corrected_images.extend(new_output.corrected_images)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import dask.array as da

class IlluminationCorrection(IndependentStepClass):
    def __init__(self, da, sigma_dict, display_plots=False):
        """
        Initialize the IlluminationCorrection class.

        Parameters:
        - da: Dask array with shape [p, t, c, y, x]
        - sigma_dict: Dictionary with sigma values per channel {channel_index: sigma_value}
        - display_plots: Boolean to control plotting
        """
        self.da = da
        self.sigma_dict = sigma_dict
        self.display_plots = display_plots

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
        """Compute the averaged illumination profile for a single channel across all images."""
        avg_projection = None
        list_images = da.shape[0]
        num_images = len(da.shape[0])

        for image in list_images:
            if image.ndim < 5 or channel >= image.shape[2]:
                print(f"Warning: Skipping image with incompatible dimensions for channel {channel}")
                continue

            # Compute mean over positions and time
            projection = image.mean(axis=(0, 1))[channel].compute()
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

    def correct_image(self, image, smoothed_profiles):
        """Apply the calculated illumination correction to each channel independently."""
        epsilon = 1e-6
        if image.ndim < 5:
            raise ValueError("Image must have at least 5 dimensions [p, t, c, y, x]")

        corrected_image = image.copy()

        for c in range(image.shape[2]):
            if c not in smoothed_profiles:
                print(f"Warning: No illumination profile for channel {c}. Skipping correction for this channel.")
                continue

            correction_factor = 1.0 / (smoothed_profiles[c] + epsilon)
            correction_factor /= np.median(correction_factor)

            # Expand correction_factor to match image dimensions
            correction_factor = correction_factor[np.newaxis, np.newaxis, np.newaxis, :, :]

            # Multiply image by correction factor
            corrected_channel = corrected_image[:, :, c, :, :] * correction_factor

            # Rescale intensity
            min_intensity = corrected_channel.min().compute()
            max_intensity = corrected_channel.max().compute()
            corrected_channel = exposure.rescale_intensity(
                corrected_channel.compute(), out_range=(min_intensity, max_intensity)
            )

            # Assign back to corrected_image
            corrected_image[:, :, c, :, :] = da.from_array(corrected_channel)

        return corrected_image

    def process_images(self, list_images):
        """
        Main function to process the images.

        Parameters:
        - list_images: List of dask arrays with shape [p, t, c, y, x]

        Returns:
        - corrected_images: List of corrected images in the same format as input
        """
        # Step 1: Compute averaged and smoothed illumination profiles for each channel across all images
        averaged_profiles = {}
        smoothed_profiles = {}
        for channel, sigma in self.sigma_dict.items():
            print(f"Calculating averaged and smoothed illumination profile for channel {channel} with sigma={sigma}...")
            avg_profile, smoothed_profile = self.average_illumination_profile(list_images, channel, sigma_smooth=sigma)
            averaged_profiles[channel] = avg_profile
            smoothed_profiles[channel] = smoothed_profile

            if self.display_plots:
                self.show_smoothed_profile(avg_profile, smoothed_profile, channel)

        # Step 2: Apply correction to each image using the smoothed profiles
        corrected_images = []
        for idx, image in enumerate(list_images):
            print(f"Correcting image {idx + 1}/{len(list_images)}...")
            corrected_image = self.correct_image(image, smoothed_profiles)
            corrected_images.append(corrected_image)

            if self.display_plots and idx == 0:
                self.show_corrected_max_projection(image, corrected_image)

        return corrected_images

    def show_smoothed_profile(self, avg_profile, smoothed_profile, channel):
        """Display averaged and smoothed illumination profiles for a channel."""
        plt.ioff()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        sns.heatmap(avg_profile, cmap='hot', cbar=True, ax=axes[0])
        axes[0].set_title(f'Averaged Profile - Channel {channel}')
        axes[0].axis('off')
        sns.heatmap(smoothed_profile, cmap='hot', cbar=True, ax=axes[1])
        axes[1].set_title(f'Smoothed Profile - Channel {channel}')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def show_corrected_max_projection(self, original_image, corrected_image):
        """Display max projections of the original and corrected images for all channels."""
        num_channels = original_image.shape[2]

        for channel in range(num_channels):
            # Compute mean over positions and time
            original_proj = original_image[:, :, channel, :, :].mean(axis=(0, 1)).compute()
            corrected_proj = corrected_image[:, :, channel, :, :].mean(axis=(0, 1)).compute()

            # Rescale intensities for visualization
            original_proj_rescaled = exposure.rescale_intensity(
                original_proj, in_range=(np.percentile(original_proj, 1), np.percentile(original_proj, 99))
            )
            corrected_proj_rescaled = exposure.rescale_intensity(
                corrected_proj, in_range=(np.percentile(corrected_proj, 1), np.percentile(corrected_proj, 99))
            )

            # Plotting
            plt.ioff()
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
            axes[0].imshow(original_proj_rescaled, cmap='hot')
            axes[0].set_title(f'Original Projection - Channel {channel}')
            axes[0].axis('off')

            axes[1].imshow(corrected_proj_rescaled, cmap='hot')
            axes[1].set_title(f'Corrected Projection - Channel {channel}')
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