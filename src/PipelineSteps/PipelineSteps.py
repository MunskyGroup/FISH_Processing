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

import torch
import warnings
# import tensorflow as tf

import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot

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
        self.df_cellresults = pd.concat([self.df_cellresults, newOutput.df_cellresults])
        self.df_spotresults = pd.concat([self.df_spotresults, newOutput.df_spotresults])
        self.df_clusterresults = pd.concat([self.df_clusterresults, newOutput.df_clusterresults])


class BIGFISH_SpotDetection(PipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, masks_nuclei, masks_complete_cells, FISHChannel,  nucChannel,
             voxel_size_yx, voxel_size_z, image_name,
             spot_yx, spot_z, map_id_imgprops, bigfish_mean_threshold:float = None, bigfish_alpha: float = 0.7, bigfish_beta:float = 1,
             bigfish_gamma:float = 5, CLUSTER_RADIUS:float= 500,
             MIN_NUM_SPOT_FOR_CLUSTER:int = 4, use_log_hook:bool = False, verbose:bool = False, **kwargs):
        
        nuc_label = masks_nuclei[id]
        cell_label = masks_complete_cells[id]
        img = list_images[id]
        img = img.astype('uint16')
        self.image_name = image_name
        for c in range(len(FISHChannel)):
            rna = img[:, :, :, FISHChannel[c]]
            threshold = bigfish_mean_threshold[c]

            nuc = img[:, :, :, nucChannel[0]]

            cell_label = cell_label.astype('uint16')
            nuc_label = nuc_label.astype('uint16')

            spots, dense_regions, reference_spot, clusters = self.detect_spots(
                rna, voxel_size_yx, voxel_size_z, spot_yx, spot_z, bigfish_alpha,
                bigfish_beta, bigfish_gamma, CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER, threshold, use_log_hook, verbose
            )

            df = self.extract_cell_level_results(spots, clusters, nuc_label, cell_label, rna, nuc, verbose)

            df['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df)
            df['fov'] = [map_id_imgprops[id]['fov_num']]*len(df)
            df['FISH_Channel'] = [c]*len(df)

            df_spotresults = pd.DataFrame(spots, columns=['z', 'y', 'x', 'cluster_index'])
            df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
            df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
            df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

            df_clusterresults = pd.DataFrame(clusters, columns=['z', 'y', 'x', 'nb_spots', 'cluster_index'])
            df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
            df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
            df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

            output = BIGFISH_SpotOutputClass(img_id=id, df_cellresults=df, df_spotresults=df_spotresults, df_clusterresults=df_clusterresults)
            return output


    def detect_spots(self, rna, voxel_size_yx, voxel_size_z, spot_yx, spot_z, alpha, beta, gamma,
                     CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER, threshold, use_log_hook, verbose):
        voxel_size = (voxel_size_z, voxel_size_yx, voxel_size_yx)
        spot_size = (spot_z, spot_yx, spot_yx)

        if use_log_hook:
            spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=spot_size,
                ndim=3)
            if verbose:
                print("spot radius (z axis): {:0.3f} pixels".format(spot_radius_px[0]))
                print("spot radius (yx plan): {:0.3f} pixels".format(spot_radius_px[-1]))

            spot_size = (spot_radius_px[0], spot_radius_px[-1], spot_radius_px[-1])

            spots, threshold = detection.detect_spots(
                images=rna,
                threshold=threshold,
                return_threshold=True,
                log_kernel_size=spot_size,
                minimum_distance=spot_size)
            if verbose:
                print("detected spots")
                print("\r shape: {0}".format(spots.shape))
                print("\r threshold: {0}".format(threshold))
            if self.step_output_dir is not None:
                plot.plot_elbow(
                    images=rna,
                    minimum_distance=spot_size,
                    log_kernel_size=spot_size, 
                    path_output=os.path.join(self.step_output_dir, f'elbowPlot_{self.image_name}'))

        else:
            spot_size = spot_size
            spots, threshold = detection.detect_spots(
                images=rna,
                threshold=threshold,
                return_threshold=True,
                voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
                spot_radius=spot_size)  # in nanometer (one value per dimension zyx)
            if verbose:
                print("detected spots")
                print("\r shape: {0}".format(spots.shape))
                print("\r threshold: {0}".format(threshold))
            if self.step_output_dir is not None:
                plot.plot_elbow(
                    images=rna,
                    voxel_size=voxel_size,
                    spot_radius=spot_size,
                    path_output=os.path.join(self.step_output_dir, f'elbowPlot_{self.image_name}'))
        if self.step_output_dir is not None:
            plot.plot_detection(np.max(rna, axis=0), spots, contrast=True, 
                                path_output=os.path.join(self.step_output_dir, f'detection_{self.image_name}'))

        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
            image=rna,
            spots=spots,
            voxel_size=voxel_size,
            spot_radius=spot_size,
            alpha=alpha,  # alpha impacts the number of spots per candidate region
            beta=beta,  # beta impacts the number of candidate regions to decompose
            gamma=gamma,)  # gamma the filtering step to denoise the image
        if verbose:
            print("detected spots before decomposition")
            print("\r shape: {0}".format(spots.shape))
            print("detected spots after decomposition")
            print("\r shape: {0}".format(spots_post_decomposition.shape))

        if self.step_output_dir is not None:
            if self.step_output_dir is not None:
                plot.plot_reference_spot(reference_spot, rescale=True, 
                                         path_output=os.path.join(self.step_output_dir, f'referenceSpot_{self.image_name}'))

        spots_post_clustering, clusters = detection.detect_clusters(
            spots=spots_post_decomposition,
            voxel_size=voxel_size,
            radius=CLUSTER_RADIUS,
            nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
        if verbose:
            print("detected spots after clustering")
            print("\r shape: {0}".format(spots_post_clustering.shape))
            print("detected clusters")
            print("\r shape: {0}".format(clusters.shape))

        return spots_post_clustering, dense_regions, reference_spot, clusters

    def extract_cell_level_results(self, spots, clusters, nuc_label, cell_label, rna, nuc, verbose):
        spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_label, ndim=3)
        if verbose:
            print("detected spots (without transcription sites)")
            print("\r shape: {0}".format(spots_no_ts.shape))
            print("\r dtype: {0}".format(spots_no_ts.dtype))

        spots_in, spots_out = multistack.identify_objects_in_region(nuc_label, spots, ndim=3)
        if verbose:
            print("detected spots (inside nuclei)")
            print("\r shape: {0}".format(spots_in.shape))
            print("\r dtype: {0}".format(spots_in.dtype), "\n")
            print("detected spots (outside nuclei)")
            print("\r shape: {0}".format(spots_out.shape))
            print("\r dtype: {0}".format(spots_out.dtype))

        fov_results = multistack.extract_cell(
            cell_label=cell_label,
            ndim=3,
            nuc_label=nuc_label,
            rna_coord=spots_no_ts,
            others_coord={"foci": foci, "transcription_site": ts},
            image=np.max(rna, axis=0),
            others_image={"dapi": np.max(nuc, axis=0)},)
        if verbose:
            print("number of cells identified: {0}".format(len(fov_results)))

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

            # plot cell
            if self.step_output_dir is not None:
                plot.plot_cell(
                    ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord,
                    rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord,
                    image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask,
                    title="Cell {0}".format(i), path_output=os.path.join(self.step_output_dir, f'cell_{self.image_name}_cell{i}'))

        df = multistack.summarize_extraction_results(fov_results, ndim=3)

        return df

    def standarize_output(self):
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

