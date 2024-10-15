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
# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, StepOutputsClass, SingleStepCompiler

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection

class filter_output(StepOutputsClass):
    def __init__(self, image: np.array):
        super().__init__()
        self.ModifyPipelineData = True
        self.list_images = [image]

    def append(self, new_output):
        self.list_images = [*self.list_images, *new_output.list_images]

# class filter_output(StepOutputsClass):
#     def __init__(self, image: np.array):
#         super().__init__()
#         self.ModifyPipelineData = True
#         self.corrected_images = [image]

#     def append(self, new_output):
#         self.corrected_images = [*self.corrected_images, *new_output.corrected_images]


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


class illumination_correction(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, list_images: list, FISHChannel, cytoChannel, nucChannel, sigma: float = 50, output_dir: str = None, save_images: bool = False,
            display_plots: bool = True, show_final_projection: bool = True, show_illumination_profile: bool = True, max_images: int = 5, **kwargs):
        """Perform exposure correction on multiple images using nuclear and cyto channels."""
        corrected_images = []

        # Limit the number of images to process
        if max_images is not None:
            list_images = list_images[:max_images]
            print(f"Limiting the number of images to process to {max_images}")

        # Ensure the output directory exists if save_images is True
        if save_images:
            if output_dir is None:
                raise ValueError("An output directory must be specified when 'save_images' is set to True.")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # Step 1: Compute the average illumination profile across all images
        print("Calculating average illumination profile...")
        illumination_profile = self.average_illumination_profile(list_images, cytoChannel, nucChannel, sigma)

        if show_illumination_profile:
            # Display the illumination profile as a heatmap
            self.show_illumination_profile(illumination_profile)

        # Step 2: Correct each image based on the illumination profile
        print("Starting image correction...")
        for idx, image in enumerate(list_images):
            print(f"Processing image {idx + 1}/{len(list_images)}...")
            # Make a copy of the original image to avoid modifying it
            corrected_image = image.copy()

            # Correct the 3D FISH channels using the averaged illumination profile
            corrected_image = self.correct_image(corrected_image, FISHChannel, illumination_profile, display_plots=display_plots)

            if show_final_projection:
                # Generate a 2D max projection of the corrected 3D stack for visualization
                self.show_corrected_max_projection(image, corrected_image, FISHChannel)

            corrected_images.append(corrected_image)
            
            if save_images:
                # Generate a unique filename for the corrected image
                unique_filename = os.path.join(output_dir, f'corrected_image_{idx}.tif')
                
                try:
                    # Save the corrected image as a 3D TIFF stack
                    imsave(unique_filename, corrected_image, plugin='tifffile')
                    print(f"Corrected 3D image saved: {unique_filename}")
                except Exception as e:
                    print(f"Failed to save image {idx}: {e}")
                    continue

        # Explicitly exit the function after processing all images
        print("Processing complete. Exiting the function.")
        return corrected_images


    def average_illumination_profile(self, list_images, cytoChannel, nucChannel, sigma):
        """Compute the averaged illumination profile across all images."""
        avg_cyto_projection = None
        avg_nuc_projection = None
        num_images = len(list_images)

        # Iterate over each image to calculate the max projection and sum them up
        for image in list_images:
            cyto_projection = np.max(image[:, :, :, cytoChannel], axis=0)
            nuc_projection = np.max(image[:, :, :, nucChannel], axis=0)
            
            # Initialize or accumulate the projections
            if avg_cyto_projection is None:
                avg_cyto_projection = cyto_projection
                avg_nuc_projection = nuc_projection
            else:
                avg_cyto_projection += cyto_projection
                avg_nuc_projection += nuc_projection

        # Average the projections
        avg_cyto_projection /= num_images
        avg_nuc_projection /= num_images

        # Estimate the illumination profile based on the averaged projections
        illumination_profile = (avg_cyto_projection + avg_nuc_projection) / 2

        # Apply Gaussian filter to smooth the illumination profile
        illumination_profile_smooth = gaussian_filter(illumination_profile, sigma=sigma)

        # Normalize the illumination profile so that its maximum value is 1
        illumination_profile_smooth = illumination_profile_smooth / np.max(illumination_profile_smooth)

        return illumination_profile_smooth

    def correct_image(self, image, FISHChannel, illumination_profile, display_plots: bool = False):
        """Apply the estimated illumination correction to the entire 3D FISH stack."""
        # Remove extra dimensions from the illumination profile if necessary
        illumination_profile = np.squeeze(illumination_profile)

        # Add a small value to the illumination profile to prevent division by small numbers
        epsilon = 1e-6  # Small regularization factor to avoid division by zero or small numbers
        illumination_profile = illumination_profile + epsilon

        # Compute the inverse of the illumination profile for brightening
        correction_factor = 1.0 / illumination_profile

        # Iterate over each FISH channel
        for f in FISHChannel:  # FISH channels to be corrected
            # Correct each z-slice in the FISH channel
            for z in range(image.shape[0]):  # Iterate over z-stack
                rna_slice = np.squeeze(image[z, :, :, f])

                # Step 1: Apply the correction factor to brighten dim regions
                rna_corrected = rna_slice * correction_factor

                # Step 2: Rescale intensity to a standard range (optional)
                rna_corrected = exposure.rescale_intensity(rna_corrected, out_range=(0, 1))

                # Step 3: Apply adaptive histogram equalization (CLAHE) (optional)
                rna_corrected = exposure.equalize_adapthist(rna_corrected)

                # Rescale back to the original intensity range
                image[z, :, :, f] = exposure.rescale_intensity(
                    rna_corrected, out_range=(np.min(image[z, :, :, f]), np.max(image[z, :, :, f]))
                )

        return image

    def show_corrected_max_projection(self, original_image, corrected_image, FISHChannel):
        """Display the max projection of the corrected 3D FISH stack alongside the original."""
        for f in FISHChannel:
            # Original max projection
            original_max_projection = np.max(original_image[:, :, :, f], axis=0)
            # Corrected max projection
            corrected_max_projection = np.max(corrected_image[:, :, :, f], axis=0)

            # Display side-by-side before and after
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_max_projection, cmap='gray')
            axes[0].set_title(f'Original max projection for channel {f}')
            axes[1].imshow(corrected_max_projection, cmap='gray')
            axes[1].set_title(f'Corrected max projection for channel {f}')
            plt.show()

    def show_illumination_profile(self, illumination_profile):
        """Display the reconstructed illumination profile as a heatmap."""
        # Squeeze to ensure it's 2D
        illumination_profile_2d = np.squeeze(illumination_profile)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(illumination_profile_2d, cmap='viridis', cbar=True)
        plt.title('Reconstructed Illumination Profile')
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