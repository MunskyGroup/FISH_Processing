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


class SpotDetectionStepClass(PipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self,
             id: int,
             list_images: list[np.array],
             cytoChannel: list[int],
             nucChannel: list[int],
             FISHChannel: list[int],
             list_image_names: list,
             temp_folder_name: str | pathlib.Path,
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
             filtered_folder_name: str | pathlib.Path,
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
