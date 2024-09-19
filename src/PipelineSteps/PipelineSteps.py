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

from src import PipelineStepsClass, StepOutputsClass

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection


#%% Luis Steps
# _____________ Cell Segmentation Steps _____________
class CellSegmentationOutput(StepOutputsClass):
    def __init__(self, masks_complete_cells, masks_nuclei, masks_cytosol, segmentation_successful,
                 number_detected_cells, id):
        super().__init__()
        self.step_name = 'CellSegmentation'
        self.masks_complete_cells = [masks_complete_cells]
        self.masks_nuclei = [masks_nuclei]
        self.masks_cytosol = [masks_cytosol]
        self.segmentation_successful = [segmentation_successful]
        self.number_detected_cells = [number_detected_cells]
        self.img_id = [id]

    def append(self, newOutputs):
        self.masks_complete_cells = self.masks_complete_cells + newOutputs.masks_complete_cells
        self.masks_nuclei = self.masks_nuclei + newOutputs.masks_nuclei
        self.masks_cytosol = self.masks_cytosol + newOutputs.masks_cytosol
        self.segmentation_successful = self.segmentation_successful + newOutputs.segmentation_successful
        self.number_detected_cells = self.number_detected_cells + newOutputs.number_detected_cells
        self.img_id = self.img_id + newOutputs.img_id


class CellSegmentationStepClass(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.save_masks_as_file = None
        self.nucChannel = None
        self.cytoChannel = None
        self.temp_file_name = None
        self.number_detected_cells = None
        self.segmentation_successful = None
        self.masks_cytosol_no_nuclei = None
        self.ModifyPipelineData = True

    def main(self,
             save_masks_as_file,
             save_files,
             diameter_cytosol,
             diameter_nucleus,
             show_plots,
             optimization_segmentation_method,
             NUMBER_OF_CORES,
             model_nuc_segmentation,
             model_cyto_segmentation,
             pretrained_model_nuc_segmentation,
             pretrained_model_cyto_segmentation,
             MINIMAL_NUMBER_OF_PIXELS_IN_MASK,
             remove_z_slices_borders,
             NUMBER_Z_SLICES_TO_TRIM,
             initial_data_location,
             cytoChannel,
             nucChannel,
             list_image_names,
             temp_folder_name,
             list_images,
             local_mask_folder,
             list_metric_sharpness_images,
             list_is_image_sharp,
             id: int = None, **kwargs,
             ) -> CellSegmentationOutput:

        name_for_files = initial_data_location.name
        file_name = list_image_names[id]
        img = list_images[id]
        self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK = MINIMAL_NUMBER_OF_PIXELS_IN_MASK
        self.local_mask_folder = local_mask_folder
        self.cytoChannel = cytoChannel
        self.nucChannel = nucChannel
        self.save_masks_as_file = save_masks_as_file
        self.save_files = save_files



        try: # from sharpness calc
            sharpness_metric = list_metric_sharpness_images[id]
            is_image_sharp = list_is_image_sharp[id]
        except AttributeError:
            print('Skipping Sharpness Filtering')


        if save_masks_as_file and save_files:
            self.masks_folder_name = str('masks_' + name_for_files)
            if not os.path.exists(self.masks_folder_name):
                os.makedirs(self.masks_folder_name)

        # slicing the name of the file. Removing after finding '.' in the string.
        self.temp_file_name = file_name[:file_name.rfind('.')]
        temp_original_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                         'ori_' + self.temp_file_name + '.png')

        if save_files:
            Plots().plot_images(img, figsize=(15, 10), image_name=temp_original_img_name,
                                show_plots=show_plots)

        segmentation_successful = None

        if not is_image_sharp and is_image_sharp is not None:
            print('    Image out of focus.')
            segmentation_successful = False

        if not segmentation_successful and segmentation_successful is not None:
            raise Exception('Segmentation Pre-Emptively Failed')
        else:
            # Cell segmentation
            temp_segmentation_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                            'seg_' + self.temp_file_name + '.png')
            # print('- CELL SEGMENTATION')
            if local_mask_folder is None:
                self.masks_complete_cells, self.masks_nuclei, self.masks_cytosol_no_nuclei = (
                    CellSegmentation(img,
                                     cytoChannel,
                                     nucChannel,
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
                                     pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation).calculate_masks())

            else:
                self.load_in_already_made_masks()

            segmentation_successful, number_detected_cells = self.test_segementation()

            if not segmentation_successful:
                number_detected_cells = 0

            self.save_img_results()

            # OUTPUTS
            single_image_cell_segmentation_output = (
                CellSegmentationOutput(masks_complete_cells=self.masks_complete_cells,
                                       masks_nuclei=self.masks_nuclei,
                                       masks_cytosol=self.masks_cytosol_no_nuclei,
                                       segmentation_successful=segmentation_successful,
                                       number_detected_cells=self.number_detected_cells,
                                       id=id))

            return single_image_cell_segmentation_output

    def test_segementation(self):
        # test if segmentation was successful
        if self.cytoChannel is None:
            detected_mask_pixels = np.count_nonzero([self.masks_nuclei.flatten()])
            self.number_detected_cells = np.max(self.masks_nuclei)
        if self.nucChannel is None:
            detected_mask_pixels = np.count_nonzero([self.masks_complete_cells.flatten()])
            self.number_detected_cells = np.max(self.masks_complete_cells)
        if self.nucChannel is not None and self.cytoChannel is not None:
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
        if self.save_masks_as_file and self.segmentation_successful and self.save_files:
            self.number_detected_cells = np.max(self.masks_complete_cells)
            print('    Number of detected cells:                ', self.number_detected_cells)
            if self.nucChannel is not None:
                mask_nuc_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                   'masks_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_nuc_path, self.masks_nuclei)
            if self.cytoChannel is not None:
                mask_cyto_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                    'masks_cyto_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_path, self.masks_complete_cells)
            if self.cytoChannel is not None and self.nucChannel is not None:
                mask_cyto_no_nuclei_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                              'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_no_nuclei_path, self.masks_cytosol_no_nuclei)

    def load_in_already_made_masks(self):
        # Paths to masks
        if not Utilities().is_None(self.nucChannel):
            mask_nuc_path = self.local_mask_folder.absolute().joinpath('masks_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.masks_nuclei = imread(str(mask_nuc_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_nuclei)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing nuc masks')
        if not Utilities().is_None(self.cytoChannel):
            mask_cyto_path = self.local_mask_folder.absolute().joinpath('masks_cyto_' + self.temp_file_name + '.tif')
            try:
                self.masks_complete_cells = imread(str(mask_cyto_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_complete_cells)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto masks.')
        if ((not Utilities().is_None(self.nucChannel)) and
                (not Utilities().is_None(self.cytoChannel))):
            self.mask_cyto_no_nuclei_path = self.local_mask_folder.absolute().joinpath(
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


# _____________ Spot Detection Steps _____________
class SpotDetectionStepOutputClass(StepOutputsClass):
    def __init__(self, id, individual_threshold_spot_detection, avg_number_of_spots_per_cell_each_ch, dfFISH):
        self.individual_threshold_spot_detection = [individual_threshold_spot_detection]
        self.img_id = [id]
        self.avg_number_of_spots_per_cell_each_ch = [avg_number_of_spots_per_cell_each_ch]
        self.dfFISH = dfFISH

    def append(self, newOutputs):
        self.individual_threshold_spot_detection = self.individual_threshold_spot_detection + newOutputs.individual_threshold_spot_detection
        self.img_id = self.img_id + newOutputs.img_id
        self.avg_number_of_spots_per_cell_each_ch = self.avg_number_of_spots_per_cell_each_ch + newOutputs.avg_number_of_spots_per_cell_each_ch
        self.dfFISH = newOutputs.dfFISH  # I believe it does this in place :(


class SpotDetectionStepClass_Luis(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self,
             id: int,
             list_images: list[np.array],
             cytoChannel: list[int],
             nucChannel: list[int],
             FISHChannel: list[int],
             list_image_names: list,
             temp_folder_name: Union[str, pathlib.Path],
             threshold_for_spot_detection: float,
             segmentation_successful: list[bool],
             CLUSTER_RADIUS: float,
             minimum_spots_cluster: float,
             masks_complete_cells: list[np.array],
             masks_nuclei: list[np.array],
             masks_cytosol: list[np.array],
             voxel_size_z: float,
             voxel_size_yx: float,
             psf_z: float,
             psf_yx: float,
             save_all_images: bool,
             filtered_folder_name: Union[str, pathlib.Path],
             show_plots: bool = True,
             display_spots_on_multiple_z_planes: bool = False,
             use_log_filter_for_spot_detection: bool = False,
             save_files: bool = True,
             **kwargs) -> SpotDetectionStepOutputClass:

        img = list_images[id]
        masks_complete_cells = masks_complete_cells[id]
        masks_nuclei = masks_nuclei[id]
        masks_cytosol = masks_cytosol[id]
        file_name = list_image_names[id]
        temp_folder_name = temp_folder_name
        segmentation_successful = segmentation_successful[id]
        list_voxels = [voxel_size_z, voxel_size_yx]
        list_psfs = [psf_z, psf_yx]

        temp_file_name = file_name[:file_name.rfind(
            '.')]  # slicing the name of the file. Removing after finding '.' in the string.
        temp_original_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                    'ori_' + temp_file_name + '.png')

        temp_segmentation_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                        'seg_' + temp_file_name + '.png')
        # Modified Luis's Code
        if segmentation_successful:
            temp_detection_img_name = pathlib.Path().absolute().joinpath(temp_folder_name, 'det_' + temp_file_name)
            dataframe_FISH, list_fish_images, thresholds_spot_detection = (
                SpotDetection(img,
                              FISHChannel,
                              cytoChannel,
                              nucChannel,
                              cluster_radius=CLUSTER_RADIUS,
                              minimum_spots_cluster=minimum_spots_cluster,
                              masks_complete_cells=masks_complete_cells,
                              masks_nuclei=masks_nuclei,
                              masks_cytosol_no_nuclei=masks_cytosol,
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
                              save_files=save_files,
                              **kwargs).get_dataframe())

            self.dataframe = dataframe_FISH
            # print(dataframe)

            print('    Intensity threshold for spot detection : ', str(thresholds_spot_detection))
            # Create the image with labels.
            df_test = self.dataframe.loc[self.dataframe['image_id'] == id]
            test_cells_ids = np.unique(df_test['cell_id'].values)
            # Saving the average number of spots per cell
            list_number_of_spots_per_cell_for_each_spot_type = []
            list_max_number_of_spots_per_cell_for_each_spot_type = []
            for sp in range(len(FISHChannel)):
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
            if save_all_images:
                for j in range(len(FISHChannel)):
                    filtered_image_path = pathlib.Path().absolute().joinpath(filtered_folder_name, 'filter_Ch_' + str(
                        FISHChannel[j]) + '_' + temp_file_name + '.tif')
                    tifffile.imwrite(filtered_image_path, list_fish_images[j])
            # Create the image with labels.
            df_subset = dataframe_FISH.loc[dataframe_FISH['image_id'] == id]
            df_labels = df_subset.drop_duplicates(subset=['cell_id'])
            # Plotting cells 
            if save_files:
                Plots().plotting_masks_and_original_image(image=img,
                                                          masks_complete_cells=masks_complete_cells,
                                                          masks_nuclei=masks_nuclei,
                                                          channels_with_cytosol=cytoChannel,
                                                          channels_with_nucleus=nucChannel,
                                                          image_name=temp_segmentation_img_name,
                                                          show_plots=show_plots,
                                                          df_labels=df_labels)
            # del masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, list_fish_images,df_subset,df_labels
        else:
            raise Exception('Segmentation was not successful, so spot detection was not performed.')

        # OUTPUTS:
        output = SpotDetectionStepOutputClass(id=id,
                                              individual_threshold_spot_detection=thresholds_spot_detection,
                                              avg_number_of_spots_per_cell_each_ch=list_number_of_spots_per_cell_for_each_spot_type,
                                              dfFISH=self.dataframe)
        return output

    def first_run(self, id):
        self.dataframe = None


#%% Jacks Steps
class BIGFISH_Tensorflow_Segmentation(PipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, nucChannel, cytoChannel, segmentation_smoothness: int = 7,
             watershed_threshold:int = 500, watershed_alpha:float = 0.9, bigfish_targetsize:int = 256, verbose:bool = False, **kwargs):
        raise Exception('This code has not been implemented due to needing to change tensor flow version')
    
        # img = list_images[id]
        # nuc = img[:, :, :, nucChannel[0]]
        # cell = img[:, :, :, cytoChannel[0]]
        #
        # model_nuc = segmentation.unet_3_classes_nuc()
        #
        # nuc_label = segmentation.apply_unet_3_classes(
        #     model_nuc, nuc, target_size=bigfish_targetsize, test_time_augmentation=True)
        #
        # # plot.plot_segmentation(nuc, nuc_label, rescale=True)
        #
        # # apply watershed
        # cell_label = segmentation.cell_watershed(cell, nuc_label, threshold=watershed_threshold, alpha=watershed_alpha)
        #
        # # plot.plot_segmentation_boundary(cell, cell_label, nuc_label, contrast=True, boundary_size=4)
        #
        # nuc_label = segmentation.clean_segmentation(nuc_label, delimit_instance=True)
        # cell_label = segmentation.clean_segmentation(cell_label, smoothness=segmentation_smoothness, delimit_instance=True)
        # nuc_label, cell_label = multistack.match_nuc_cell(nuc_label, cell_label, single_nuc=False, cell_alone=True)
        #
        # plot.plot_images([nuc_label, cell_label], titles=["Labelled nuclei", "Labelled cells"])
        #
        # mask_cytosol = cell_label.copy()
        # mask_cytosol[nuc_label > 0 and cell_label > 0] = 0
        #
        # num_of_cells = np.max(nuc_label)
        #
        # output = CellSegmentationOutput(masks_complete_cells=cell_label, masks_nuclei=nuc_label,
        #                                 masks_cytosol=mask_cytosol, segmentation_successful=1,
        #                                 number_detected_cells=num_of_cells, id=id)
        # return output


class BIGFISH_SpotOutputClass(StepOutputsClass):
    def __init__(self, img_id, df_cellresults, df_spotresults, df_clusterresults):
        self.img_id = [img_id]
        self.df_cellresults = df_cellresults
        self.df_spotresults = df_spotresults
        self.df_clusterresults = df_clusterresults

    def append(self, newOutput):
        self.img_id = [*self.img_id, *newOutput.img_id]
    
        if self.df_cellresults is None:
            self.df_cellresults = newOutput.df_cellresults
        else:
            self.df_cellresults = pd.concat([self.df_cellresults, newOutput.df_cellresults])

        self.df_spotresults = pd.concat([self.df_spotresults, newOutput.df_spotresults])
        self.df_clusterresults = pd.concat([self.df_clusterresults, newOutput.df_clusterresults])


class BIGFISH_SpotDetection(PipelineStepsClass):
    """
    A class for detecting RNA spots in FISH images using the BIGFISH library.
    Methods
    -------
    __init__():
        Initializes the BIGFISH_SpotDetection class.
    main(id, list_images, FISHChannel, nucChannel, voxel_size_yx, voxel_size_z, spot_yx, spot_z, map_id_imgprops, image_name=None, masks_nuclei=None, masks_complete_cells=None, bigfish_mean_threshold=None, bigfish_alpha=0.7, bigfish_beta=1, bigfish_gamma=5, CLUSTER_RADIUS=500, MIN_NUM_SPOT_FOR_CLUSTER=4, use_log_hook=False, verbose=False, display_plots=False, **kwargs):
        Main method to detect spots in FISH images and extract cell-level results.
        Parameters:
        - id (int): Identifier for the image.
        - list_images (list): List of images.
        - FISHChannel (list): List of FISH channels.
        - nucChannel (list): List of nuclear channels.
        - voxel_size_yx (float): Voxel size in the yx plane.
        - voxel_size_z (float): Voxel size in the z plane.
        - spot_yx (float): Spot size in the yx plane.
        - spot_z (float): Spot size in the z plane.
        - map_id_imgprops (dict): Mapping of image properties.
        - image_name (str, optional): Name of the image.
        - masks_nuclei (list[np.array], optional): List of nuclear masks.
        - masks_complete_cells (list[np.array], optional): List of complete cell masks.
        - bigfish_mean_threshold (list[float], optional): List of mean thresholds for spot detection.
        - bigfish_alpha (float, optional): Alpha parameter for spot decomposition.
        - bigfish_beta (float, optional): Beta parameter for spot decomposition.
        - bigfish_gamma (float, optional): Gamma parameter for spot decomposition.
        - CLUSTER_RADIUS (float, optional): Radius for clustering spots.
        - MIN_NUM_SPOT_FOR_CLUSTER (int, optional): Minimum number of spots for clustering.
        - use_log_hook (bool, optional): Whether to use log kernel for spot detection.
        - verbose (bool, optional): Whether to print verbose output.
        - display_plots (bool, optional): Whether to display plots.
    """
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, FISHChannel,  nucChannel,
             voxel_size_yx, voxel_size_z, 
             spot_yx, spot_z, map_id_imgprops, 
             image_name: str = None, masks_nuclei: list[np.array] = None, masks_complete_cells: list[np.array] = None,
             bigfish_mean_threshold:list[float] = None, bigfish_alpha: float = 0.7, bigfish_beta:float = 1,
             bigfish_gamma:float = 5, CLUSTER_RADIUS:int = 500,
             MIN_NUM_SPOT_FOR_CLUSTER:int = 4, use_log_hook:bool = False, 
             verbose:bool = False, display_plots: bool = False,
              sub_pixel_fitting: bool = False, **kwargs):
        print(map_id_imgprops)
        
        # Load in images and masks
        nuc_label = masks_nuclei[id] if masks_nuclei is not None else None
        cell_label = masks_complete_cells[id] if masks_complete_cells is not None else None
        img = list_images[id]
        self.image_name = image_name

        # cycle through FISH channels
        for c in range(len(FISHChannel)):
            # extract single rna channel
            rna = img[:, :, :, FISHChannel[c]]
            nuc = img[:, :, :, nucChannel[0]]

            # check if a threshold is provided
            if bigfish_mean_threshold is not None:
                threshold = bigfish_mean_threshold[c]
            else:
                threshold = None

            # detect spots
            spots_px, dense_regions, reference_spot, clusters, spots_subpx = self.bigfish_spotdetection(
                rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=bigfish_alpha,
                beta=bigfish_beta, gamma=bigfish_gamma, CLUSTER_RADIUS=CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER=MIN_NUM_SPOT_FOR_CLUSTER, 
                threshold=threshold, use_log_hook=use_log_hook, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting)

            # extract cell level results
            if nuc_label is not None or cell_label is not None:
                df = self.extract_cell_level_results(spots_px, clusters, nuc_label, cell_label, rna, nuc, 
                                                 verbose, display_plots)
                df['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df)
                df['fov'] = [map_id_imgprops[id]['fov_num']]*len(df)
                df['FISH_Channel'] = [c]*len(df)

            else:
                df = None

            # merge spots_px and spots_um
            if spots_px.shape[1] == 4:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    
                    df_spotresults = pd.DataFrame(spots, columns=['z_px', 'y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['z_px', 'y_px', 'x_px', 'cluster_index'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
            
            else:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    
                    df_spotresults = pd.DataFrame(spots, columns=['y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['y_px', 'x_px', 'cluster_index'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

            # create output object
            output = BIGFISH_SpotOutputClass(img_id=id, df_cellresults=df, df_spotresults=df_spotresults, df_clusterresults=df_clusterresults)
            return output


    def bigfish_spotdetection(self, rna:np.array, voxel_size_yx:float, voxel_size_z:float, spot_yx:float, spot_z:float, alpha:int, beta:int,
                               gamma:int, CLUSTER_RADIUS:float, MIN_NUM_SPOT_FOR_CLUSTER:int, threshold:float, use_log_hook:bool, 
                               verbose: bool = False, display_plots: bool = False, sub_pixel_fitting: bool = False,):
        rna = rna.squeeze()

        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))

        if use_log_hook:
            spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size_nm, 
                object_radius_nm=voxel_size_nm, 
                ndim=3 if len(rna.shape) == 3 else 2)
            

        canidate_spots = detection.detect_spots(
                                            images=rna, 
                                            return_threshold=False, 
                                            threshold=threshold,
                                            voxel_size=voxel_size_nm if not use_log_hook else None,
                                            spot_radius=spot_size_nm if not use_log_hook else None,
                                            log_kernel_size=spot_radius_px if use_log_hook else None,
                                            minimum_distance=spot_radius_px if use_log_hook else None,)
        
        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                image=rna.astype(np.uint16), 
                                                spots=canidate_spots, 
                                                voxel_size=voxel_size_nm, 
                                                spot_radius=spot_size_nm,
                                                alpha=alpha,
                                                beta=beta,
                                                gamma=gamma)
        
        spots_post_clustering, clusters = detection.detect_clusters(
                                                        spots=spots_post_decomposition, 
                                                        voxel_size=voxel_size_nm, 
                                                        radius=CLUSTER_RADIUS, 
                                                        nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
        
        if sub_pixel_fitting:
            spots_subpx = detection.fit_subpixel(
                                        image=rna, 
                                        spots=canidate_spots, 
                                        voxel_size=voxel_size_nm, 
                                        spot_radius=voxel_size_nm)
        else:
            spots_subpx = None
            
        if verbose:
            print("detected canidate spots")
            print("\r shape: {0}".format(canidate_spots.shape))
            print("\r threshold: {0}".format(threshold))
            print("detected spots after decomposition")
            print("\r shape: {0}".format(spots_post_decomposition.shape))
            print("detected spots after clustering")
            print("\r shape: {0}".format(spots_post_clustering.shape))
            print("detected clusters")
            print("\r shape: {0}".format(clusters.shape))

        if display_plots:
            plot.plot_elbow(
                images=rna, 
                voxel_size=voxel_size_nm if not use_log_hook else None, 
                spot_radius=spot_size_nm if not use_log_hook else None,
                log_kernel_size=spot_radius_px if use_log_hook else None,
                minimum_distance=spot_radius_px if use_log_hook else None,
                path_output=os.path.join(self.step_output_dir, f'elbow_{self.image_name}') if self.step_output_dir is not None else None)
            plot.plot_reference_spot(reference_spot, rescale=True, 
                                    path_output=os.path.join(self.step_output_dir, f'reference_spot_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    canidate_spots, contrast=True, 
                                    path_output=os.path.join(self.step_output_dir, f'canidate_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    spots_post_decomposition, contrast=True, 
                                    path_output=os.path.join(self.step_output_dir, f'detection_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0), 
                                    spots=[spots_post_decomposition, clusters[:, :2] if len(rna.shape) == 2 else clusters[:, :3]], 
                                    shape=["circle", "circle"], 
                                    radius=[3, 6], 
                                    color=["red", "blue"],
                                    linewidth=[1, 2], 
                                    fill=[False, True], 
                                    contrast=True,
                                    path_output=os.path.join(self.step_output_dir, f'cluster_{self.image_name}') if self.step_output_dir is not None else None)
        return spots_post_clustering, dense_regions, reference_spot, clusters, spots_subpx

    def extract_cell_level_results(self, spots, clusters, nuc_label, cell_label, rna, nuc, verbose, display_plots):
        # convert masks to max projection
        if nuc_label is not None and len(nuc_label.shape) != 2:
            nuc_label = np.max(nuc_label, axis=0)
        if cell_label is not None and len(cell_label.shape) != 2:
            cell_label = np.max(cell_label, axis=0)

        # remove transcription sites
        spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_label, ndim=3)
        if verbose:
            print("detected spots (without transcription sites)")
            print("\r shape: {0}".format(spots_no_ts.shape))
            print("\r dtype: {0}".format(spots_no_ts.dtype))

        # get spots inside and outside nuclei
        spots_in, spots_out = multistack.identify_objects_in_region(nuc_label, spots, ndim=3)
        if verbose:
            print("detected spots (inside nuclei)")
            print("\r shape: {0}".format(spots_in.shape))
            print("\r dtype: {0}".format(spots_in.dtype), "\n")
            print("detected spots (outside nuclei)")
            print("\r shape: {0}".format(spots_out.shape))
            print("\r dtype: {0}".format(spots_out.dtype))

        # extract fov results
        other_images = {}
        other_images["dapi"] = np.max(nuc, axis=0).astype("uint16") if nuc is not None else None
        fov_results = multistack.extract_cell(
            cell_label=cell_label.astype("uint16") if cell_label is not None else nuc_label.astype("uint16"),
            ndim=3,
            nuc_label=nuc_label.astype("uint16"),
            rna_coord=spots_no_ts,
            others_coord={"foci": foci, "transcription_site": ts},
            image=np.max(rna, axis=0).astype("uint16"),
            others_image=other_images,)
        if verbose:
            print("number of cells identified: {0}".format(len(fov_results)))

        # cycle through cells and save the results
        for i, cell_results in enumerate(fov_results):
            # get cell results
            cell_mask = cell_results["cell_mask"]
            cell_coord = cell_results["cell_coord"]
            nuc_mask = cell_results["nuc_mask"]
            nuc_coord = cell_results["nuc_coord"]
            rna_coord = cell_results["rna_coord"]
            foci_coord = cell_results["foci"]
            ts_coord = cell_results["transcription_site"]
            image_contrasted = cell_results["image"]
            if verbose:
                print("cell {0}".format(i))
                print("\r number of rna {0}".format(len(rna_coord)))
                print("\r number of foci {0}".format(len(foci_coord)))
                print("\r number of transcription sites {0}".format(len(ts_coord)))

            # plot individual cells
            if display_plots:
                plot.plot_cell(
                    ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord,
                    rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord,
                    image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask,
                    title="Cell {0}".format(i), 
                    path_output=os.path.join(self.step_output_dir, f'cell_{self.image_name}_cell{i}') if self.step_output_dir is not None else None)

        df = multistack.summarize_extraction_results(fov_results, ndim=3)
        return df

    def get_spot_properties(self, rna, spot, voxel_size_yx, voxel_size_z, spot_yx, spot_z):
        pass

class CellSegmentationStepClass_JF(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.save_masks_as_file = None
        self.nucChannel = None
        self.cytoChannel = None
        self.temp_file_name = None
        self.number_detected_cells = None
        self.segmentation_successful = None
        self.masks_cytosol_no_nuclei = None
        self.ModifyPipelineData = True

    def main(self,
             save_masks_as_file,
             save_files,
             diameter_cytosol,
             diameter_nucleus,
             show_plots,
             optimization_segmentation_method,
             NUMBER_OF_CORES,
             model_nuc_segmentation,
             model_cyto_segmentation,
             pretrained_model_nuc_segmentation,
             pretrained_model_cyto_segmentation,
             MINIMAL_NUMBER_OF_PIXELS_IN_MASK,
             initial_data_location,
             cytoChannel,
             nucChannel,
             list_image_names,
             list_images,
             local_mask_folder,
             list_metric_sharpness_images: list[float] = None,
             list_is_image_sharp: list[bool] = None,
             id: int = None, **kwargs,
             ) -> CellSegmentationOutput:

        name_for_files = initial_data_location.name
        image_name = list_image_names[id]
        img = list_images[id]
        self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK = MINIMAL_NUMBER_OF_PIXELS_IN_MASK
        self.local_mask_folder = local_mask_folder
        self.cytoChannel = cytoChannel
        self.nucChannel = nucChannel
        self.save_masks_as_file = save_masks_as_file
        self.save_files = save_files
        image_name = os.path.splitext(image_name)[0]
        self.temp_file_name = image_name

        try: # from sharpness calc
            sharpness_metric = list_metric_sharpness_images[id]
            is_image_sharp = list_is_image_sharp[id]
        except TypeError:
            print('Skipping Sharpness Filtering')
            is_image_sharp = None

        if save_masks_as_file and save_files:
            self.masks_folder_name = str('masks_' + name_for_files)
            if not os.path.exists(self.masks_folder_name):
                os.makedirs(self.masks_folder_name)

        if save_files:
            Plots().plot_images(img, figsize=(15, 10), image_name=os.path.join(self.step_output_dir, 'ori_' + image_name + '.png'),
                                show_plots=show_plots)

        segmentation_successful = None

        if not is_image_sharp and is_image_sharp is not None:
            print('    Image out of focus.')
            segmentation_successful = False

        if not segmentation_successful and segmentation_successful is not None:
            raise Exception('Segmentation Pre-Emptively Failed')
        else:
            if local_mask_folder is None:
                self.masks_complete_cells, self.masks_nuclei, self.masks_cytosol_no_nuclei = (
                    CellSegmentation(img,
                                     cytoChannel,
                                     nucChannel,
                                     diameter_cytosol=diameter_cytosol,
                                     diameter_nucleus=diameter_nucleus,
                                     show_plots=show_plots,
                                     optimization_segmentation_method=optimization_segmentation_method,
                                     image_name=os.path.join(self.step_output_dir, 'seg_' + image_name + '.png'),
                                     NUMBER_OF_CORES=NUMBER_OF_CORES,
                                     running_in_pipeline=True,
                                     model_nuc_segmentation=model_nuc_segmentation,
                                     model_cyto_segmentation=model_cyto_segmentation,
                                     pretrained_model_nuc_segmentation=pretrained_model_nuc_segmentation,
                                     pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation).calculate_masks())

            else:
                self.load_in_already_made_masks()

            segmentation_successful, number_detected_cells = self.test_segementation()

            if not segmentation_successful:
                number_detected_cells = 0

            self.save_img_results()

            # OUTPUTS
            single_image_cell_segmentation_output = (
                CellSegmentationOutput(masks_complete_cells=self.masks_complete_cells,
                                       masks_nuclei=self.masks_nuclei,
                                       masks_cytosol=self.masks_cytosol_no_nuclei,
                                       segmentation_successful=segmentation_successful,
                                       number_detected_cells=self.number_detected_cells,
                                       id=id))

            return single_image_cell_segmentation_output

    def test_segementation(self):
        # test if segmentation was successful
        if self.cytoChannel is None:
            detected_mask_pixels = np.count_nonzero([self.masks_nuclei.flatten()])
            self.number_detected_cells = np.max(self.masks_nuclei)
        if self.nucChannel is None:
            detected_mask_pixels = np.count_nonzero([self.masks_complete_cells.flatten()])
            self.number_detected_cells = np.max(self.masks_complete_cells)
        if self.nucChannel is not None and self.cytoChannel is not None:
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
        if self.save_masks_as_file and self.segmentation_successful and self.save_files:
            self.number_detected_cells = np.max(self.masks_complete_cells)
            print('    Number of detected cells:                ', self.number_detected_cells)
            if self.nucChannel is not None:
                mask_nuc_path = os.path.join(self.step_output_dir,
                                                                   'masksNuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_nuc_path, self.masks_nuclei)
            if self.cytoChannel is not None:
                mask_cyto_path = os.path.join(self.step_output_dir,
                                                                    'masksCyto_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_path, self.masks_complete_cells)
            if self.cytoChannel is not None and self.nucChannel is not None:
                mask_cyto_no_nuclei_path = os.path.join(self.step_output_dir,
                                                                              'masksCytoNoNuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_no_nuclei_path, self.masks_cytosol_no_nuclei)

    def load_in_already_made_masks(self):
        # Paths to masks
        if not Utilities().is_None(self.nucChannel):
            mask_nuc_path = self.local_mask_folder.absolute().joinpath('masks_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.masks_nuclei = imread(str(mask_nuc_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_nuclei)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing nuc masks')
        if not Utilities().is_None(self.cytoChannel):
            mask_cyto_path = self.local_mask_folder.absolute().joinpath('masks_cyto_' + self.temp_file_name + '.tif')
            try:
                self.masks_complete_cells = imread(str(mask_cyto_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.masks_complete_cells)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto masks.')
        if ((not Utilities().is_None(self.nucChannel)) and
                (not Utilities().is_None(self.cytoChannel))):
            self.mask_cyto_no_nuclei_path = self.local_mask_folder.absolute().joinpath(
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


class SimpleCellposeSegmentaion(PipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, cytoChannel, nucChannel, image_name: str = None,
             cellpose_model_type: str = 'cyto3', cellpose_diameter: float = 70, cellpose_channel_axis: int = 3,
             cellpose_invert: bool = False, cellpose_normalize: bool = True, cellpose_do_3D: bool = False, 
             cellpose_min_size: float = None, cellpose_flow_threshold: float = 0.3,
             display_plots: bool = False,
               **kwargs):
        if cellpose_min_size is None:
            cellpose_min_size = np.pi * (cellpose_diameter / 4) ** 2

        if not cellpose_do_3D:
            image = np.max(image, axis=0)
        
        nuc_channel = nucChannel[0] if nucChannel is not None else 0
        cyto_channel = cytoChannel[0] if cytoChannel is not None else 0
        channels = [[cyto_channel, nuc_channel]]

        model = models.Cellpose(model_type=cellpose_model_type, gpu=True)
        masks, flows, styles, diams = model.eval(image, channels=channels, diameter=cellpose_diameter, 
                                                 invert=cellpose_invert, normalize=cellpose_normalize, 
                                                 channel_axis=cellpose_channel_axis, do_3D=cellpose_do_3D,
                                                 min_size=cellpose_min_size, flow_threshold=cellpose_flow_threshold)
        
        if nucChannel is not None and len(masks.shape) == cellpose_channel_axis+1:
            nuc_mask = np.take(masks, nuc_channel, axis=cellpose_channel_axis) if len(masks.shape) == cellpose_channel_axis+1 and nucChannel is not None else masks
        elif nucChannel is not None:
            nuc_mask = masks
        else:
            nuc_mask = None

        if cytoChannel is not None and len(masks.shape) == cellpose_channel_axis+1:
            cell_mask = np.take(masks, cyto_channel, axis=cellpose_channel_axis) if len(masks.shape) == cellpose_channel_axis+1 and cytoChannel is not None else masks
        elif cytoChannel is not None:
            cell_mask = masks
        else:
            cell_mask = None

        cyto_mask = cell_mask[nuc_mask>0] if cell_mask is not None and nuc_mask is not None else None

        number_detected_cells = np.max(cell_mask) if cell_mask is not None else np.max(nuc_mask)

        if display_plots:
            num_sub_plots = 1
            if nuc_mask is not None:
                num_sub_plots += 2
            if cyto_mask is not None:
                num_sub_plots += 2
            fig, axs = plt.subplots(1, num_sub_plots, figsize=(12, 5))
            i = 0
            if nuc_mask is not None:
                if cellpose_do_3D:
                    axs[i].imshow(np.max(image,axis=0)[:, : nucChannel[0]], cmap='gray')
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(np.max(nuc_mask,axis=0), cmap='tab20')
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[:,:,nucChannel[0]], cmap='gray')
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(nuc_mask, cmap='tab20')
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
            if cell_mask is not None:
                if cellpose_do_3D:
                    axs[i].imshow(np.max(image,axis=0)[:, : cytoChannel[0]], cmap='gray')
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(np.max(cell_mask, axis=0), cmap='tab20')
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[:,:,cytoChannel[0]], cmap='gray')
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(cell_mask, cmap='tab20')
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                if cellpose_do_3D:
                    axs[i].imshow(flows[0][image.shape[0]//2, :, :])
                    axs[i].set_title('Flow')
                else:
                    axs[i].imshow(flows[0][:, :])
                    axs[i].set_title('Flow')
            plt.tight_layout()
            if self.step_output_dir is not None:
                plt.savefig(os.path.join(self.step_output_dir, f'{image_name}_segmentation.png'))
            plt.show()


        return CellSegmentationOutput(masks_complete_cells=cell_mask,
                                        masks_nuclei=nuc_mask,
                                        masks_cytosol=cyto_mask,
                                        segmentation_successful=1,
                                        number_detected_cells=number_detected_cells,
                                        id=0)


# %% Parameter Optimization Steps
class ParamOptimizer_BIGFISH_SpotDetection_Output(StepOutputsClass):
    def __init__(self, array_of_outputs, order_of_index: dict):
        super().__init__()
        self.array_of_outputs = array_of_outputs
        self.order_of_index = order_of_index

    def append(self, newOutput):
        # for each element in array_of_outputs, append the corresponding element in newOutput.array_of_outputs
        for i, output in np.ndenumerate(self.array_of_outputs):
            output.append(newOutput.array_of_outputs[i]) #  This is awesome cause it mutable 

        

class ParamOptimizer_BIGFISH_SpotDetection(PipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, bigfish_mean_threshold: list[float] = [None], bigfish_alpha: list[float] = [0.7], bigfish_beta: list[float] = [1],
             bigfish_gamma: list[float] = [5], CLUSTER_RADIUS: list[float] = [500],
             MIN_NUM_SPOT_FOR_CLUSTER: list[int] = [4], **kwargs):
        
        self.mean_threshold = bigfish_mean_threshold if type(bigfish_mean_threshold) is list else [bigfish_mean_threshold]
        self.alpha = bigfish_alpha if type(bigfish_alpha) is list else [bigfish_alpha]
        self.beta = bigfish_beta if type(bigfish_beta) is list else [bigfish_beta]
        self.gamma = bigfish_gamma  if type(bigfish_gamma) is list else [bigfish_gamma]
        self.cluster_radius = CLUSTER_RADIUS if type(CLUSTER_RADIUS) is list else [CLUSTER_RADIUS]
        self.min_num_spot_for_cluster = MIN_NUM_SPOT_FOR_CLUSTER if type(MIN_NUM_SPOT_FOR_CLUSTER) is list else [MIN_NUM_SPOT_FOR_CLUSTER]

        order_of_index = {'mean_threshold': 0, 'alpha': 1, 'beta': 2, 'gamma': 3, 'cluster_radius': 4, 'min_num_spot_for_cluster': 5}
        output = np.empty((len(self.mean_threshold), len(self.alpha), len(self.beta), len(self.gamma), len(self.cluster_radius), len(self.min_num_spot_for_cluster)), dtype=object)
        for i, threshold in enumerate(self.mean_threshold):
            for j, alpha in enumerate(self.alpha):
                for k, beta in enumerate(self.beta):
                    for l, gamma in enumerate(self.gamma):
                        for m, radius in enumerate(self.cluster_radius):
                            for n, min_num in enumerate(self.min_num_spot_for_cluster):
                                print(f'Running threshold: {threshold}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, cluster_radius: {radius}, min_num: {min_num}')
                                kwargs['bigfish_mean_threshold'] = threshold
                                kwargs['bigfish_alpha'] = alpha
                                kwargs['bigfish_beta'] = beta
                                kwargs['bigfish_gamma'] = gamma
                                kwargs['CLUSTER_RADIUS'] = radius
                                kwargs['MIN_NUM_SPOT_FOR_CLUSTER'] = min_num
                                kwargs['id'] = id
                                single_output = BIGFISH_SpotDetection().main(**kwargs)
                                if output[i,j,k,l,m,n] is None:
                                    output[i,j,k,l,m,n] = single_output
                                else:
                                    output[i,j,k,l,m,n].append(single_output)

        return ParamOptimizer_BIGFISH_SpotDetection_Output(output, order_of_index)

class Trackpy_SpotDetection_Output(StepOutputsClass):
    def __init__(self, id, trackpy_features):
        super().__init__()
        self.id = [id]
        self.trackpy_features = trackpy_features

    def append(self, newOutput):
        self.id = [*self.id, *newOutput.id]
        self.trackpy_features = pd.concat([self.trackpy_features, newOutput.trackpy_features])


class TrackPy_SpotDetection(PipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, FISHChannel, spot_yx_px, spot_z_px, voxel_size_yx, voxel_size_z,
             map_id_imgprops, trackpy_minmass: float = None,  trackpy_minsignal: float = 1000, 
             trackpy_seperation_yx_px: float = 13, trackpy_seperation_z_px: float = 3,
               display_plots: bool = False, plot_types: list[str] = ['mass', 'size', 'signal', 'raw_mass'], **kwargs):
        # Load in image and extract FISH channel
        img = list_images[id]
        fish = np.squeeze(img[:, :, :, FISHChannel[0]])   

        # 3D spot detection
        if len(fish.shape) == 3:
            spot_diameter = (spot_z_px, spot_yx_px, spot_yx_px)
            separation = (trackpy_seperation_z_px, trackpy_seperation_yx_px, trackpy_seperation_yx_px) if trackpy_seperation_yx_px is not None else None
            trackpy_features = tp.locate(fish, diameter=spot_diameter, minmass=trackpy_minmass, separation=separation)

        # 2D spot detection
        else:
            spot_diameter = spot_yx_px
            separation = trackpy_seperation_yx_px
            trackpy_features = tp.locate(fish, diameter=spot_diameter, minmass=trackpy_minmass, separation=separation)

        # Plotting
        if display_plots:
            if len(fish.shape) == 3:
                tp.annotate3d(trackpy_features, fish)
                tp.subpx_bias(trackpy_features)

            else:
                tp.annotate(trackpy_features, fish)
                tp.subpx_bias(trackpy_features)
            
            for plot_type in plot_types:
                fig, ax = plt.subplots()
                ax.hist(trackpy_features[plot_type], bins=20)
                # Optionally, label the axes.
                ax.set(xlabel=plot_type, ylabel='count')
                plt.show()

        # append frame number and fov number to the features
        trackpy_features['frame'] = [map_id_imgprops[id]['tp_num']]*len(trackpy_features)
        trackpy_features['fov'] = [map_id_imgprops[id]['fov_num']]*len(trackpy_features)
        trackpy_features['FISH_Channel'] = [FISHChannel[0]]*len(trackpy_features)
        trackpy_features['xum'] = trackpy_features['x']*voxel_size_yx/1000 # convert to microns
        trackpy_features['yum'] = trackpy_features['y']*voxel_size_yx/1000
        if len(fish.shape) == 3:
            trackpy_features['zum'] = trackpy_features['z']*voxel_size_z/1000

        output = Trackpy_SpotDetection_Output(id=id, trackpy_features=trackpy_features)
        return output








