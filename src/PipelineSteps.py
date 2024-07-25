from skimage import img_as_float64, img_as_uint
from skimage.filters import gaussian
from joblib import Parallel, delayed
import multiprocessing
import bigfish.stack as stack
import bigfish.plot as plot
import bigfish.detection as detection
import bigfish.multistack as multistack
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import re
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.morphology import erosion
from scipy import ndimage
from scipy.optimize import curve_fit
import itertools
import glob
import tifffile
import sys
import datetime
import getpass
import pkg_resources
import platform
import math
from cellpose import models
import os; from os import listdir; from os.path import isfile, join
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

import torch
import warnings

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch
    number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
    if number_gpus >1 : # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  str(np.random.randint(0,number_gpus,1)[0])
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import zipfile
import seaborn as sns
import scipy.stats as stats
from  matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
font_props = {'size': 16}
import joypy
from matplotlib import cm
from scipy.ndimage import binary_dilation

from src import PipelineStepsClass, StepOutputsClass
from src.Util.Utilities import Utilities
from src.Util.Plots import Plots
from src.Util.CellSegmentation import CellSegmentation
from src.Util.SpotDetection import SpotDetection


# _____________ Cell Segmentation Steps _____________
class CellSegmentationOutput(StepOutputsClass):
    def __init__(self, masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, segmentation_successful,
                 number_detected_cells, id):
        super().__init__()
        self.step_name = 'CellSegmentation'
        self.masks_complete_cells = [masks_complete_cells]
        self.masks_nuclei = [masks_nuclei]
        self.masks_cytosol_no_nuclei = [masks_cytosol_no_nuclei]
        self.segmentation_successful = [segmentation_successful]
        self.number_detected_cells = [number_detected_cells]
        self.img_id = [id]

    def append(self, newOutputs):
        self.masks_complete_cells = self.masks_complete_cells + newOutputs.masks_complete_cells
        self.masks_nuclei = self.masks_nuclei + newOutputs.masks_nuclei
        self.masks_cytosol_no_nuclei = self.masks_cytosol_no_nuclei + newOutputs.masks_cytosol_no_nuclei
        self.segmentation_successful = self.segmentation_successful + newOutputs.segmentation_successful
        self.number_detected_cells = self.number_detected_cells + newOutputs.number_detected_cells
        self.img_id = self.img_id + newOutputs.img_id


class CellSegmentationStepClass(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK = None
        self.channels_with_nucleus = None
        self.channels_with_cytosol = None
        self.masks_dir = None
        self.save_masks_as_file = None

    def main(self, id: int, pipelineData, pipelineSettings, experiment, terminatorScope) -> CellSegmentationOutput:
        # INPUTS
        self.save_masks_as_file = pipelineSettings.save_masks_as_file
        save_files = pipelineSettings.save_files
        name_for_files = experiment.initial_data_location.name
        file_name = pipelineData.list_image_names[id]
        temp_folder_name = pipelineData.temp_folder_name
        img = pipelineData.list_images[id]
        NUMBER_Z_SLICES_TO_TRIM = pipelineSettings.NUMBER_Z_SLICES_TO_TRIM
        sharpness_metric = pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_metric_sharpeness_images[id]
        is_image_sharp = pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_is_image_sharp[id]
        self.masks_dir = pipelineData.local_mask_folder
        self.channels_with_cytosol = experiment.cytoChannel
        self.channels_with_nucleus = experiment.nucChannel
        diameter_cytosol = pipelineSettings.diameter_cytosol
        diameter_nucleus = pipelineSettings.diameter_nucleus
        show_plots = pipelineSettings.show_plots
        optimization_segmentation_method = pipelineSettings.optimization_segmentation_method
        NUMBER_OF_CORES = pipelineSettings.NUMBER_OF_CORES
        model_nuc_segmentation = pipelineSettings.model_nuc_segmentation
        model_cyto_segmentation = pipelineSettings.model_cyto_segmentation
        pretrained_model_nuc_segmentation = pipelineSettings.pretrained_model_nuc_segmentation
        pretrained_model_cyto_segmentation = pipelineSettings.pretrained_model_cyto_segmentation
        self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK = pipelineSettings.MINIMAL_NUMBER_OF_PIXELS_IN_MASK
        remove_z_slices_borders = pipelineSettings.remove_z_slices_borders

        # LUIS'S CODE
        if (self.save_masks_as_file == True) and (save_files == True):
            self.masks_folder_name = str('masks_' + name_for_files)
            if not os.path.exists(self.masks_folder_name):
                os.makedirs(self.masks_folder_name)

        self.temp_file_name = file_name[:file_name.rfind(
            '.')]  # slcing the name of the file. Removing after finding '.' in the string.
        self.temp_original_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                         'ori_' + self.temp_file_name + '.png')
        if save_files == True:
            Plots().plot_images(img, figsize=(15, 10), image_name=self.temp_original_img_name, show_plots=show_plots)
        img_shape = list(img.shape)
        if remove_z_slices_borders == True:
            img_shape = list(img.shape)
            img_shape[0] = img_shape[0] + 2 * (NUMBER_Z_SLICES_TO_TRIM)
            print('    Orginal Image Shape :                    ', img_shape)
            print('    Trimmed z_slices at each border :        ', NUMBER_Z_SLICES_TO_TRIM)
        else:
            print('    Original Image Shape :                   ', img_shape)
        print('    Image sharpness metric :                 ', sharpness_metric)

        if is_image_sharp == False:
            print('    Image out of focus.')
            self.segmentation_successful = False
        else:
            # Cell segmentation
            temp_segmentation_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                            'seg_' + self.temp_file_name + '.png')
            #print('- CELL SEGMENTATION')
            if (self.masks_dir is None):

                self.masks_complete_cells, self.masks_nuclei, self.masks_cytosol_no_nuclei = CellSegmentation(img,
                                                                                                              self.channels_with_cytosol,
                                                                                                              self.channels_with_nucleus,
                                                                                                              diameter_cytosol=diameter_cytosol,
                                                                                                              diameter_nucleus=diameter_nucleus,
                                                                                                              show_plots=show_plots,
                                                                                                              optimization_segmentation_method=optimization_segmentation_method,
                                                                                                              image_name=temp_segmentation_img_name,
                                                                                                              NUMBER_OF_CORES=NUMBER_OF_CORES,
                                                                                                              running_in_pipeline=True,
                                                                                                              model_nuc_segmentation=model_nuc_segmentation,
                                                                                                              model_cyto_segmentation=model_cyto_segmentation,
                                                                                                              pretrained_model_nuc_segmentation=pretrained_model_nuc_segmentation,
                                                                                                              pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation).calculate_masks()

            else:
                self.load_in_already_made_masks()

            self.segmentation_successful, self.number_detected_cells = self.test_segementation()

            if self.segmentation_successful == False:
                self.number_detected_cells = 0

            self.save_img_results()

            # OUTPUTS
            singleImageCellSegmentationOutput = CellSegmentationOutput(masks_complete_cells=self.masks_complete_cells,
                                                                       masks_nuclei=self.masks_nuclei,
                                                                       masks_cytosol_no_nuclei=self.masks_cytosol_no_nuclei,
                                                                       segmentation_successful=self.segmentation_successful,
                                                                       number_detected_cells=self.number_detected_cells,
                                                                       id=id)
            return singleImageCellSegmentationOutput

    def test_segementation(self):
        # test if segmentation was succcesful
        if Utilities().is_None(self.channels_with_cytosol) == True:  #(self.channels_with_cytosol is None):
            detected_mask_pixels = np.count_nonzero([self.masks_nuclei.flatten()])
            self.number_detected_cells = np.max(self.masks_nuclei)
        if Utilities().is_None(self.channels_with_nucleus) == True:  #(self.channels_with_nucleus  is None):
            detected_mask_pixels = np.count_nonzero([self.masks_complete_cells.flatten()])
            self.number_detected_cells = np.max(self.masks_complete_cells)
        if (Utilities().is_None(self.channels_with_nucleus) == False) and (Utilities().is_None(
                self.channels_with_cytosol) == False):  #not (self.channels_with_nucleus  is None) and not(self.channels_with_cytosol  is None):
            detected_mask_pixels = np.count_nonzero([self.masks_complete_cells.flatten(), self.masks_nuclei.flatten(),
                                                     self.masks_cytosol_no_nuclei.flatten()])
            self.number_detected_cells = np.max(self.masks_complete_cells)
        # Counting pixels
        if detected_mask_pixels > self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK:
            self.segmentation_successful = True
        else:
            self.segmentation_successful = False
            print('    Segmentation was not successful. Due to low number of detected pixels.')

        return self.segmentation_successful, self.number_detected_cells

    def save_img_results(self):
        # saving masks
        if (self.save_masks_as_file == True) and (self.segmentation_successful == True) and (self.save_files == True):
            self.number_detected_cells = np.max(self.masks_complete_cells)
            print('    Number of detected cells:                ', self.number_detected_cells)
            if Utilities().is_None(self.channels_with_nucleus) == False:  #not (self.channels_with_nucleus is None):
                mask_nuc_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                   'masks_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_nuc_path, self.masks_nuclei)
            if Utilities().is_None(self.channels_with_cytosol) == False:  #not (self.channels_with_cytosol is None):
                mask_cyto_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                    'masks_cyto_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_path, self.masks_complete_cells)
            if (Utilities().is_None(self.channels_with_nucleus) == False) and (Utilities().is_None(
                    self.channels_with_cytosol) == False):  #not (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None):
                mask_cyto_no_nuclei_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                              'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_no_nuclei_path, self.masks_cytosol_no_nuclei)

    def load_in_already_made_masks(self):
        # Paths to masks
        if Utilities().is_None(
                self.channels_with_nucleus) == False:  #not (self.channels_with_nucleus in (None,[None])) :
            mask_nuc_path = self.masks_dir.absolute().joinpath('masks_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.masks_nuclei = imread(str(mask_nuc_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_nuclei)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing nuc masks')
        if Utilities().is_None(self.channels_with_cytosol) == False:  #not (self.channels_with_cytosol is None):
            mask_cyto_path = self.masks_dir.absolute().joinpath('masks_cyto_' + self.temp_file_name + '.tif')
            try:
                self.masks_complete_cells = imread(str(mask_cyto_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_complete_cells)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto masks.')
        if (Utilities().is_None(self.channels_with_nucleus) == False) and (Utilities().is_None(
                self.channels_with_cytosol) == False):  # not (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None) :
            self.mask_cyto_no_nuclei_path = self.masks_dir.absolute().joinpath(
                'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.masks_cytosol_no_nuclei = imread(str(self.mask_cyto_no_nuclei_path))
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto only masks.')
        # test all masks exist, if not create the variable and set as None.
        if not 'masks_nuclei' in locals():
            self.masks_nuclei = None
        if not 'masks_complete_cells' in locals():
            self.masks_complete_cells = None
        if not 'masks_cytosol_no_nuclei' in locals():
            self.masks_cytosol_no_nuclei = None


class CellposeSegmentation(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, id: int, pipelineData, pipelineSettings, experiment, terminatorScope) -> CellSegmentationOutput:
        img = pipelineData.list_images[id]
        self.channels_with_cytosol = experiment.cytoChannel
        self.channels_with_nucleus = experiment.nucChannel

        if not self.channels_with_cytosol is None:
            cytoImgs = img[:, :, :, self.channels_with_cytosol]

        if not self.channels_with_nucleus is None:
            nucImgs = img[:, :, :, self.channels_with_nucleus]

        # Cellpose segmentation


# _____________ Spot Detection Steps _____________
class SpotDetectionStepOutputClass(StepOutputsClass):
    def __init__(self, id, threshold_spot_detection, avg_number_of_spots_per_cell_each_ch, dfFISH):
        self.thresholds_spot_detection = [threshold_spot_detection]
        self.img_id = [id]
        self.avg_number_of_spots_per_cell_each_ch = [avg_number_of_spots_per_cell_each_ch]
        self.dfFISH = dfFISH

    def append(self, newOutputs):
        self.thresholds_spot_detection = self.thresholds_spot_detection + newOutputs.thresholds_spot_detection
        self.img_id = self.img_id + newOutputs.img_id
        self.avg_number_of_spots_per_cell_each_ch = self.avg_number_of_spots_per_cell_each_ch + newOutputs.avg_number_of_spots_per_cell_each_ch
        self.dfFISH = newOutputs.dfFISH  # I beleive it does this in place :(


class SpotDetectionStepClass(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, id: int, pipelineData, pipelineSettings, experiment,
             terminatorScope) -> SpotDetectionStepOutputClass:
        # INPUTS
        self.channels_with_cytosol = experiment.cytoChannel
        self.channels_with_nucleus = experiment.nucChannel
        self.channels_with_FISH = experiment.FISHChannel
        file_name = pipelineData.list_image_names[id]
        temp_folder_name = pipelineData.temp_folder_name
        segmentation_successful = pipelineData.pipelineOutputs.CellSegmentationOutput.segmentation_successful[id]
        self.CLUSTER_RADIUS = pipelineSettings.CLUSTER_RADIUS
        self.minimum_spots_cluster = pipelineSettings.minimum_spots_cluster
        img = pipelineData.list_images[id]
        masks_complete_cells = pipelineData.pipelineOutputs.CellSegmentationOutput.masks_complete_cells[id]
        masks_nuclei = pipelineData.pipelineOutputs.CellSegmentationOutput.masks_nuclei[id]
        masks_cytosol_no_nuclei = pipelineData.pipelineOutputs.CellSegmentationOutput.masks_cytosol_no_nuclei[id]

        list_voxels = [experiment.voxel_size_z, terminatorScope.voxel_size_yx]
        list_psfs = [terminatorScope.psf_z, terminatorScope.psf_yx]
        save_all_images = pipelineSettings.save_all_images
        show_plots = pipelineSettings.show_plots
        display_spots_on_multiple_z_planes = pipelineSettings.display_spots_on_multiple_z_planes
        use_log_filter_for_spot_detection = pipelineSettings.use_log_filter_for_spot_detection
        threshold_for_spot_detection = pipelineData.prePipelineOutputs.AutomaticThresholdingOutput.threshold_for_spot_detection

        save_files = pipelineSettings.save_files

        if save_all_images:
            filtered_folder_name = pipelineData.filtered_folder_name

        temp_file_name = file_name[:file_name.rfind(
            '.')]  # slcing the name of the file. Removing after finding '.' in the string.
        temp_original_img_name = pathlib.Path().absolute().joinpath(temp_folder_name, 'ori_' + temp_file_name + '.png')

        temp_segmentation_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                        'seg_' + temp_file_name + '.png')
        # Modified Luis's Code
        if segmentation_successful == True:
            temp_detection_img_name = pathlib.Path().absolute().joinpath(temp_folder_name, 'det_' + temp_file_name)
            dataframe_FISH, list_fish_images, thresholds_spot_detection = SpotDetection(img,
                                                                                        self.channels_with_FISH,
                                                                                        self.channels_with_cytosol,
                                                                                        self.channels_with_nucleus,
                                                                                        cluster_radius=self.CLUSTER_RADIUS,
                                                                                        minimum_spots_cluster=self.minimum_spots_cluster,
                                                                                        masks_complete_cells=masks_complete_cells,
                                                                                        masks_nuclei=masks_nuclei,
                                                                                        masks_cytosol_no_nuclei=masks_cytosol_no_nuclei,
                                                                                        dataframe=self.dataframe,
                                                                                        image_counter=id,
                                                                                        list_voxels=list_voxels,
                                                                                        list_psfs=list_psfs,
                                                                                        show_plots=show_plots,
                                                                                        image_name=temp_detection_img_name,
                                                                                        save_all_images=save_all_images,
                                                                                        display_spots_on_multiple_z_planes=display_spots_on_multiple_z_planes,
                                                                                        use_log_filter_for_spot_detection=use_log_filter_for_spot_detection,
                                                                                        threshold_for_spot_detection=threshold_for_spot_detection,
                                                                                        save_files=save_files).get_dataframe()
            self.dataframe = dataframe_FISH
            # print(self.dataframe)

            print('    Intensity threshold for spot detection : ', str(thresholds_spot_detection))
            # Create the image with labels.
            df_test = self.dataframe.loc[self.dataframe['image_id'] == id]
            test_cells_ids = np.unique(df_test['cell_id'].values)
            # Saving the average number of spots per cell
            list_number_of_spots_per_cell_for_each_spot_type = []
            list_max_number_of_spots_per_cell_for_each_spot_type = []
            for sp in range(len(self.channels_with_FISH)):
                detected_spots = np.asarray([len(self.dataframe.loc[(self.dataframe['cell_id'] == cell_id_test) & (
                            self.dataframe['spot_type'] == sp) & (self.dataframe['is_cell_fragmented'] != -1)].spot_id)
                                             for i, cell_id_test in enumerate(test_cells_ids)])
                average_number_of_spots_per_cell = int(np.mean(detected_spots))
                max_number_of_spots_per_cell = int(np.max(detected_spots))
                list_number_of_spots_per_cell_for_each_spot_type.append(average_number_of_spots_per_cell)
                list_max_number_of_spots_per_cell_for_each_spot_type.append(max_number_of_spots_per_cell)
            print('    Average detected spots per cell :        ', list_number_of_spots_per_cell_for_each_spot_type)
            print('    Maximum detected spots per cell :        ', list_max_number_of_spots_per_cell_for_each_spot_type)
            #list_average_spots_per_cell.append(list_number_of_spots_per_cell_for_each_spot_type)
            # saving FISH images
            if save_all_images == True:
                for j in range(len(self.channels_with_FISH)):
                    filtered_image_path = pathlib.Path().absolute().joinpath(filtered_folder_name, 'filter_Ch_' + str(
                        self.channels_with_FISH[j]) + '_' + temp_file_name + '.tif')
                    tifffile.imwrite(filtered_image_path, list_fish_images[j])
            # Create the image with labels.
            df_subset = dataframe_FISH.loc[dataframe_FISH['image_id'] == id]
            df_labels = df_subset.drop_duplicates(subset=['cell_id'])
            # Plotting cells 
            if save_files == True:
                Plots().plotting_masks_and_original_image(image=img,
                                                          masks_complete_cells=masks_complete_cells,
                                                          masks_nuclei=masks_nuclei,
                                                          channels_with_cytosol=self.channels_with_cytosol,
                                                          channels_with_nucleus=self.channels_with_nucleus,
                                                          image_name=temp_segmentation_img_name,
                                                          show_plots=show_plots,
                                                          df_labels=df_labels)
            # del masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, list_fish_images,df_subset,df_labels
        else:
            raise Exception('Segmentation was not successful, so spot detection was not performed.')

        # OUTPUTS:
        output = SpotDetectionStepOutputClass(id=id,
                                              threshold_spot_detection=threshold_for_spot_detection,
                                              avg_number_of_spots_per_cell_each_ch=list_number_of_spots_per_cell_for_each_spot_type,
                                              dfFISH=self.dataframe)
        return output

    def first_run(self):
        self.dataframe = None
