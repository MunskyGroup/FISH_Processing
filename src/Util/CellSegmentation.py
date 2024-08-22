from joblib import Parallel, delayed
import bigfish.stack as stack
import pandas as pd
import pathlib
import numpy as np
import re
from skimage.morphology import erosion
from scipy import ndimage
from scipy.optimize import curve_fit
import glob
import tifffile
import getpass
import pkg_resources
import platform
import math
from cellpose import models
import os;
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl

mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import socket
import yaml
import shutil
from fpdf import FPDF
import gc

import torch
import warnings

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

font_props = {'size': 16}
import joypy
from scipy.ndimage import binary_dilation

# from . import Cellpose, RemoveExtrema, Plots
from src.Util.CellPose import Cellpose
from src.Util.RemoveExtrema import RemoveExtrema
from src.Util.Plots import Plots


class CellSegmentation:
    """
    This class is intended to detect cells in FISH images using `Cellpose <https://github.com/MouseLand/cellpose>`_ . This class segments the nucleus and cytosol for every cell detected in the image. The class uses optimization to generate the meta-parameters used by cellpose. For a complete description of Cellpose check the `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .

    Parameters

    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] or maximum projection with dimensions [Y, X, C].
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None.
    diameter_cytosol : int, optional
        Average cytosol size in pixels. The default is 150.
    diameter_nucleus : int, optional
        Average nucleus size in pixels. The default is 100.
    optimization_segmentation_method: str
        Method used for the segmentation. The options are: \'default\', \'intensity_segmentation\', \'z_slice_segmentation_marker\', \'gaussian_filter_segmentation\', and None.
    remove_fragmented_cells: bool, optional
        If true, it removes masks in the border of the image. The default is False.
    show_plots : bool, optional
        If true, it shows a plot with the detected masks. The default is True.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    """

    def __init__(self, image: np.ndarray, channels_with_cytosol=None, channels_with_nucleus=None,
                 diameter_cytosol: float = 150, diameter_nucleus: float = 100,
                 optimization_segmentation_method='default', remove_fragmented_cells: bool = False,
                 show_plots: bool = True, image_name=None, NUMBER_OF_CORES=1, running_in_pipeline=False,
                 model_nuc_segmentation='nuclei', model_cyto_segmentation='cyto',
                 pretrained_model_nuc_segmentation=None, pretrained_model_cyto_segmentation=None):
        self.image = image
        self.channels_with_cytosol = channels_with_cytosol
        self.channels_with_nucleus = channels_with_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.diameter_nucleus = diameter_nucleus
        self.show_plots = show_plots
        self.remove_fragmented_cells = remove_fragmented_cells
        self.image_name = image_name
        self.number_z_slices = image.shape[0]
        self.NUMBER_OPTIMIZATION_VALUES = np.min((self.number_z_slices, 8))
        self.optimization_segmentation_method = optimization_segmentation_method  # optimization_segmentation_method = 'intensity_segmentation' 'default', 'gaussian_filter_segmentation' , None
        if self.optimization_segmentation_method == 'z_slice_segmentation_marker':
            self.NUMBER_OPTIMIZATION_VALUES = self.number_z_slices
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.running_in_pipeline = running_in_pipeline
        self.model_nuc_segmentation = model_nuc_segmentation
        self.model_cyto_segmentation = model_cyto_segmentation
        self.pretrained_model_nuc_segmentation = pretrained_model_nuc_segmentation
        self.pretrained_model_cyto_segmentation = pretrained_model_cyto_segmentation

    def calculate_masks(self):
        '''
        This method performs the process of cell detection for FISH images using **Cellpose**.

        Returns

        masks_complete_cells : NumPy array. np.uint8
            Image containing the masks for every cell detected in the image. Numpy array with format [Y, X].
        masks_nuclei : NumPy array. np.uint8
            Image containing the masks for every nuclei detected in the image. Numpy array with format [Y, X].
        masks_cytosol_no_nuclei : NumPy array. np.uint8
            Image containing the masks for every cytosol (removing the nucleus) detected in the image. Numpy array with format [Y, X].
        '''

        # function that determines if the nucleus is in the cytosol
        def is_nucleus_in_cytosol(mask_n, mask_c):
            mask_n[mask_n > 1] = 1
            mask_c[mask_c > 1] = 1
            size_mask_n = np.count_nonzero(mask_n)
            size_mask_c = np.count_nonzero(mask_c)
            min_size = np.min((size_mask_n, size_mask_c))
            mask_combined = mask_n + mask_c
            sum_mask = np.count_nonzero(mask_combined[mask_combined == 2])
            if (sum_mask > min_size * 0.8) and (
                    min_size > 200):  # the element is inside if the two masks overlap over the 80% of the smaller mask.
                return 1
            else:
                return 0

        ##### IMPLEMENTATION #####
        if len(self.image.shape) > 3:  # [ZYXC]
            if self.image.shape[0] == 1:
                max_image = self.image[0, :, :, :]
            else:
                center_slice = self.number_z_slices // 2
                max_image = np.max(self.image[:, :, :, :], axis=0)  # taking the mean value
                max_image = RemoveExtrema(max_image, min_percentile=1, max_percentile=98).remove_outliers()
        else:
            max_image = self.image  # [YXC]
            max_image = RemoveExtrema(max_image, min_percentile=1, max_percentile=98).remove_outliers()

            # Function to calculate the approximated radius in a mask

        def approximated_radius(masks, diameter=100):
            n_masks = np.max(masks)
            if n_masks > 1:  # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range(1,
                                n_masks + 1):  # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    approximated_radius = np.sqrt(np.sum(masks == nm) / np.pi)  # a=  pi r2
                    size_mask.append(approximated_radius)  # creating a list with the size of each mask
                array_radii = np.array(size_mask)
            else:
                array_radii = np.sqrt(np.sum(masks == 1) / np.pi)
            if np.any(array_radii > diameter * 1.5) | np.any(array_radii < 10):
                masks_radii = 0
            else:
                masks_radii = np.prod(array_radii)
            return masks_radii

        def metric_paired_masks(masks_complete_cells, masks_nuclei):
            median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,
                                                                             diameter=self.diameter_cytosol)
            median_radii_nuclei = approximated_radius(masks_nuclei, diameter=self.diameter_nucleus)
            return median_radii_nuclei * median_radii_complete_cells_complete_cells

        def metric_max_cells_and_area(masks):
            n_masks = np.max(masks)
            if n_masks > 1:  # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range(1,
                                n_masks + 1):  # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    approximated_radius = np.sqrt(np.sum(masks == nm) / np.pi)  # a=  pi r2
                    size_mask.append(
                        approximated_radius)  # np.sum(masks == nm)) # creating a list with the size of each mask
                size_masks_array = np.array(size_mask)
                metric = np.mean(size_masks_array).astype(int) * n_masks
            elif n_masks == 1:  # do nothing if only a single mask is detected per image.
                approximated_radius = np.sqrt(np.sum(masks == 1) / np.pi)
                metric = approximated_radius.astype(int)
            else:  # return zero if no mask are detected
                metric = 0
            return metric

            # Function to find masks

        def function_to_find_masks(image):
            if not (self.channels_with_cytosol in (None, [None])):
                masks_cyto = Cellpose(image[:, :, self.channels_with_cytosol], diameter=self.diameter_cytosol,
                                      model_type=self.model_cyto_segmentation, selection_method='max_cells_and_area',
                                      NUMBER_OF_CORES=self.NUMBER_OF_CORES,
                                      pretrained_model=self.pretrained_model_cyto_segmentation).calculate_masks()
            else:
                masks_cyto = np.zeros_like(image[:, :, 0])
            if not (self.channels_with_nucleus in (None, [None])):
                masks_nuclei = Cellpose(image[:, :, self.channels_with_nucleus], diameter=self.diameter_nucleus,
                                        model_type=self.model_nuc_segmentation, selection_method='max_cells_and_area',
                                        NUMBER_OF_CORES=self.NUMBER_OF_CORES,
                                        pretrained_model=self.pretrained_model_nuc_segmentation).calculate_masks()
            else:
                masks_nuclei = np.zeros_like(image[:, :, 0])
            if not (self.channels_with_cytosol in (None, [None])) and not (
                    self.channels_with_nucleus in (None, [None])):
                # Function that removes masks that are not paired with a nucleus or cytosol
                def remove_lonely_masks(masks_0, masks_1, is_nuc=None):
                    n_mask_0 = np.max(masks_0)
                    n_mask_1 = np.max(masks_1)
                    if (n_mask_0 > 0) and (n_mask_1 > 0):
                        for ind_0 in range(1, n_mask_0 + 1):
                            tested_mask_0 = erosion(np.where(masks_0 == ind_0, 1, 0))
                            array_paired = np.zeros(n_mask_1)
                            for ind_1 in range(1, n_mask_1 + 1):
                                tested_mask_1 = erosion(np.where(masks_1 == ind_1, 1, 0))
                                array_paired[ind_1 - 1] = is_nucleus_in_cytosol(tested_mask_1, tested_mask_0)
                                if (is_nuc == 'nuc') and (
                                        np.count_nonzero(tested_mask_0) > np.count_nonzero(tested_mask_1)):
                                    # condition that rejects images with nucleus bigger than the cytosol
                                    array_paired[ind_1 - 1] = 0
                                elif (is_nuc is None) and (
                                        np.count_nonzero(tested_mask_1) > np.count_nonzero(tested_mask_0)):
                                    array_paired[ind_1 - 1] = 0
                            if any(array_paired) == False:  # If the cytosol is not associated with any mask.
                                masks_0 = np.where(masks_0 == ind_0, 0, masks_0)
                            masks_with_pairs = masks_0
                    else:
                        masks_with_pairs = np.zeros_like(masks_0)
                    return masks_with_pairs

                # Function that reorder the index to make it continuos
                def reorder_masks(mask_tested):
                    n_mask_0 = np.max(mask_tested)
                    mask_new = np.zeros_like(mask_tested)
                    if n_mask_0 > 0:
                        counter = 0
                        for ind_0 in range(1, n_mask_0 + 1):
                            if ind_0 in mask_tested:
                                counter = counter + 1
                                if counter == 1:
                                    mask_new = np.where(mask_tested == ind_0, -counter, mask_tested)
                                else:
                                    mask_new = np.where(mask_new == ind_0, -counter, mask_new)
                        reordered_mask = np.absolute(mask_new)
                    else:
                        reordered_mask = mask_new
                    return reordered_mask
                    # Cytosol masks

                masks_cyto = remove_lonely_masks(masks_cyto, masks_nuclei)
                masks_cyto = reorder_masks(masks_cyto)
                # Masks nucleus
                masks_nuclei = remove_lonely_masks(masks_nuclei, masks_cyto, is_nuc='nuc')
                masks_nuclei = reorder_masks(masks_nuclei)

                # Iterate for each cyto mask
                def matching_masks(masks_cyto, masks_nuclei):
                    n_mask_cyto = np.max(masks_cyto)
                    n_mask_nuc = np.max(masks_nuclei)
                    new_masks_nuclei = np.zeros_like(masks_cyto)
                    reordered_mask_nuclei = np.zeros_like(masks_cyto)
                    if (n_mask_cyto > 0) and (n_mask_nuc > 0):
                        for mc in range(1, n_mask_cyto + 1):
                            tested_mask_cyto = np.where(masks_cyto == mc, 1, 0)
                            for mn in range(1, n_mask_nuc + 1):
                                mask_paired = False
                                tested_mask_nuc = np.where(masks_nuclei == mn, 1, 0)
                                mask_paired = is_nucleus_in_cytosol(tested_mask_nuc, tested_mask_cyto)
                                if mask_paired == True:
                                    if np.count_nonzero(new_masks_nuclei) == 0:
                                        new_masks_nuclei = np.where(masks_nuclei == mn, -mc, masks_nuclei)
                                    else:
                                        new_masks_nuclei = np.where(new_masks_nuclei == mn, -mc, new_masks_nuclei)
                            reordered_mask_nuclei = np.absolute(new_masks_nuclei)
                    else:
                        masks_cyto = np.zeros_like(masks_cyto)
                        reordered_mask_nuclei = np.zeros_like(masks_nuclei)
                    return masks_cyto, reordered_mask_nuclei

                # Matching nuclei and cytosol
                masks_cyto, masks_nuclei = matching_masks(masks_cyto, masks_nuclei)
                # Generating mask for cyto without nuc
                masks_cytosol_no_nuclei = masks_cyto - masks_nuclei
                masks_cytosol_no_nuclei[masks_cytosol_no_nuclei < 0] = 0
                masks_cytosol_no_nuclei.astype(int)
                # Renaming
                masks_complete_cells = masks_cyto
            else:
                if not (self.channels_with_cytosol in (None, [None])):
                    masks_complete_cells = masks_cyto
                    masks_nuclei = None
                    masks_cytosol_no_nuclei = None
                if not (self.channels_with_nucleus in (None, [None])):
                    masks_complete_cells = masks_nuclei
                    masks_cytosol_no_nuclei = None
            return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei

        # TESTING IF A MASK PATH WAS PASSED TO THE CODE
        # OPTIMIZATION METHODS FOR SEGMENTATION
        if (self.optimization_segmentation_method == 'intensity_segmentation') and (len(self.image.shape) > 3) and (
                self.image.shape[0] > 1):
            # Intensity Based Optimization to find the maximum number of index_paired_masks.
            sigma_value_for_gaussian = 2
            list_masks_complete_cells = []
            list_masks_nuclei = []
            list_masks_cytosol_no_nuclei = []
            if not (self.channels_with_cytosol in (None, [None])) or not (self.channels_with_nucleus in (None, [None])):
                tested_thresholds = np.round(np.linspace(0, 5, self.NUMBER_OPTIMIZATION_VALUES), 0)
                # list_sorting_number_paired_masks = []
                array_number_paired_masks = np.zeros(len(tested_thresholds))
                for idx, threshold in enumerate(tested_thresholds):
                    image_temp = RemoveExtrema(max_image, min_percentile=0.5,
                                               max_percentile=100 - threshold).remove_outliers()
                    image_filtered = stack.gaussian_filter(image_temp,
                                                           sigma=sigma_value_for_gaussian)  # max_image.copy()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(image_filtered)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = metric_max_cells_and_area(masks_complete_cells)
                    array_number_paired_masks[idx] = metric
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei[selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei[selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")

        elif (self.optimization_segmentation_method == 'z_slice_segmentation_marker') and (
                len(self.image.shape) > 3) and (self.image.shape[0] > 1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks.
            list_idx = np.round(np.linspace(0, self.number_z_slices - 1, self.NUMBER_OPTIMIZATION_VALUES), 0).astype(
                int)
            list_idx = np.unique(list_idx)  # list(set(list_idx))
            # Optimization based on slice
            if not (self.channels_with_cytosol in (None, [None])) or not (self.channels_with_nucleus in (None, [None])):
                # list_sorting_number_paired_masks = []
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros(len(list_idx) + 1)
                # performing the segmentation on a maximum projection
                test_image_optimization = np.max(self.image[:, :, :, :], axis=0)
                test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                        max_percentile=98).remove_outliers()
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                    test_image_optimization)
                list_masks_complete_cells.append(masks_complete_cells)
                list_masks_nuclei.append(masks_nuclei)
                list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                metric = metric_max_cells_and_area(masks_complete_cells)  # np.max(masks_complete_cells)
                array_number_paired_masks[0] = metric
                for idx, idx_value in enumerate(list_idx):
                    test_image_optimization = self.image[idx_value, :, :, :]
                    test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                            max_percentile=98).remove_outliers()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                        test_image_optimization)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = metric_max_cells_and_area(masks_complete_cells)  # np.max(masks_complete_cells)
                    array_number_paired_masks[idx + 1] = metric
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei[selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei[selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")

        elif (self.optimization_segmentation_method == 'default') and (len(self.image.shape) > 3) and (
                self.image.shape[0] > 1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks.
            if self.number_z_slices > 20:
                num_slices_range = 3  # range to consider above and below a selected z-slice
            else:
                num_slices_range = 1
            list_idx = np.round(
                np.linspace(num_slices_range, self.number_z_slices - num_slices_range, self.NUMBER_OPTIMIZATION_VALUES),
                0).astype(int)
            list_idx = np.unique(list_idx)  # list(set(list_idx))
            # Optimization based on slice
            if not (self.channels_with_cytosol in (None, [None])) or not (self.channels_with_nucleus in (None, [None])):
                # list_sorting_number_paired_masks = []
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros(len(list_idx) + 1)
                # performing the segmentation on a maximum projection
                test_image_optimization = np.max(self.image[num_slices_range:-num_slices_range, :, :, :], axis=0)
                test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                        max_percentile=98).remove_outliers()
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                    test_image_optimization)
                list_masks_complete_cells.append(masks_complete_cells)
                list_masks_nuclei.append(masks_nuclei)
                list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                metric = metric_max_cells_and_area(
                    masks_complete_cells)  # metric_paired_masks(masks_complete_cells,masks_nuclei)
                array_number_paired_masks[0] = metric
                # performing segmentation for a subsection of z-slices
                for idx, idx_value in enumerate(list_idx):
                    test_image_optimization = np.max(
                        self.image[idx_value - num_slices_range:idx_value + num_slices_range, :, :, :], axis=0)
                    test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                            max_percentile=98).remove_outliers()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                        test_image_optimization)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = metric_max_cells_and_area(
                        masks_complete_cells)  # metric_paired_masks(masks_complete_cells,masks_nuclei)
                    array_number_paired_masks[idx + 1] = metric
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei[selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei[selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")

        elif (self.optimization_segmentation_method == 'center_slice') and (len(self.image.shape) > 3) and (
                self.image.shape[0] > 1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks.
            center_slice = self.number_z_slices // 2
            num_slices_range = np.min((5, center_slice - 1))  # range to consider above and below a selected z-slice
            # Optimization based on slice
            test_image_optimization = self.image[center_slice, :, :, :]
            test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                    max_percentile=98).remove_outliers()
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                test_image_optimization)
        elif (self.optimization_segmentation_method == 'gaussian_filter_segmentation') and (
                len(self.image.shape) > 3) and (self.image.shape[0] > 1):
            # Optimization based on testing different sigmas in a gaussian filter to find the maximum number of index_paired_masks.
            list_sigmas = np.round(np.linspace(0.5, 4, self.NUMBER_OPTIMIZATION_VALUES), 1)
            # Optimization based on slice
            if not (self.channels_with_cytosol in (None, [None])) or not (self.channels_with_nucleus in (None, [None])):
                # list_sorting_number_paired_masks = []
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros(len(list_sigmas))
                for idx, sigma_value in enumerate(list_sigmas):
                    test_image_optimization = stack.gaussian_filter(np.max(self.image[:, :, :, :], axis=0),
                                                                    sigma=sigma_value)
                    test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=1,
                                                            max_percentile=98).remove_outliers()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(
                        test_image_optimization)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = metric_max_cells_and_area(masks_complete_cells)
                    array_number_paired_masks[idx] = metric
                # selected_threshold = list_sigmas[np.argmax(array_number_paired_masks)]
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei[selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei[selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")
        else:
            # no optimization is applied if a 2D image is passed
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(max_image)
        if self.running_in_pipeline == False:
            Plots().plotting_masks_and_original_image(image=max_image,
                                                      masks_complete_cells=masks_complete_cells,
                                                      masks_nuclei=masks_nuclei,
                                                      channels_with_cytosol=self.channels_with_cytosol,
                                                      channels_with_nucleus=self.channels_with_nucleus,
                                                      image_name=self.image_name,
                                                      show_plots=self.show_plots)

        return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei


from src.Util.Utilities import Utilities
