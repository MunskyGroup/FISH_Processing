import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src import StepOutputsClass, SequentialStepsClass

#%% Output Classes
class CellSegmentationOutput(StepOutputsClass):
    def __init__(self, list_cell_masks, list_nuc_masks, list_cyto_masks, segmentation_successful,
                 number_detected_cells, id):
        super().__init__()
        self.step_name = 'CellSegmentation'
        self.list_cell_masks = [list_cell_masks]
        self.list_nuc_masks = [list_nuc_masks]
        self.list_cyto_masks = [list_cyto_masks]
        self.segmentation_successful = [segmentation_successful]
        self.number_detected_cells = [number_detected_cells]
        self.img_id = [id]

    def append(self, newOutputs):
        self.list_cell_masks = self.list_cell_masks + newOutputs.list_cell_masks
        self.list_nuc_masks = self.list_nuc_masks + newOutputs.list_nuc_masks
        self.list_cyto_masks = self.list_cyto_masks + newOutputs.list_cyto_masks
        self.segmentation_successful = self.segmentation_successful + newOutputs.segmentation_successful
        self.number_detected_cells = self.number_detected_cells + newOutputs.number_detected_cells
        self.img_id = self.img_id + newOutputs.img_id


class CellSegmentationOutput_Luis(StepOutputsClass):
    def __init__(self, list_cell_masks, list_nuc_masks, list_cyto_masks, segmentation_successful,
                 number_detected_cells, id):
        super().__init__()
        self.step_name = 'CellSegmentation'
        self.list_cell_masks = [list_cell_masks]
        self.list_nuc_masks = [list_nuc_masks]
        self.list_cyto_masks = [list_cyto_masks]
        self.segmentation_successful = [segmentation_successful]
        self.number_detected_cells = [number_detected_cells]
        self.img_id = [id]

    def append(self, newOutputs):
        self.list_cell_masks = self.list_cell_masks + newOutputs.list_cell_masks
        self.list_nuc_masks = self.list_nuc_masks + newOutputs.list_nuc_masks
        self.list_cyto_masks = self.list_cyto_masks + newOutputs.list_cyto_masks
        self.segmentation_successful = self.segmentation_successful + newOutputs.segmentation_successful
        self.number_detected_cells = self.number_detected_cells + newOutputs.number_detected_cells
        self.img_id = self.img_id + newOutputs.img_id


#%% Steps
class SimpleCellposeSegmentaion(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, cytoChannel, nucChannel, image_name: str = None,
             cellpose_model_type: str = 'cyto3', cellpose_diameter: float = 70, cellpose_channel_axis: int = 3,
             cellpose_invert: bool = False, cellpose_normalize: bool = True, cellpose_do_3D: bool = False, 
             cellpose_min_size: float = None, cellpose_flow_threshold: float = 0.3, cellpose_cellprob_threshold: float = 0.0,
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
                                                 min_size=cellpose_min_size, flow_threshold=cellpose_flow_threshold, cellprob_threshold=cellpose_cellprob_threshold)
        
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


        return CellSegmentationOutput(list_cell_masks=cell_mask,
                                        list_nuc_masks=nuc_mask,
                                        list_cyto_masks=cyto_mask,
                                        segmentation_successful=1,
                                        number_detected_cells=number_detected_cells,
                                        id=0)
    

class CellSegmentationStepClass_JF(SequentialStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.save_masks_as_file = None
        self.nucChannel = None
        self.cytoChannel = None
        self.temp_file_name = None
        self.number_detected_cells = None
        self.segmentation_successful = None
        self.list_cyto_masks_no_nuclei = None
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
                self.list_cell_masks, self.list_nuc_masks, self.list_cyto_masks_no_nuclei = (
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
                CellSegmentationOutput(list_cell_masks=self.list_cell_masks,
                                       list_nuc_masks=self.list_nuc_masks,
                                       list_cyto_masks=self.list_cyto_masks_no_nuclei,
                                       segmentation_successful=segmentation_successful,
                                       number_detected_cells=self.number_detected_cells,
                                       id=id))

            return single_image_cell_segmentation_output

    def test_segementation(self):
        # test if segmentation was successful
        if self.cytoChannel is None:
            detected_mask_pixels = np.count_nonzero([self.list_nuc_masks.flatten()])
            self.number_detected_cells = np.max(self.list_nuc_masks)
        if self.nucChannel is None:
            detected_mask_pixels = np.count_nonzero([self.list_cell_masks.flatten()])
            self.number_detected_cells = np.max(self.list_cell_masks)
        if self.nucChannel is not None and self.cytoChannel is not None:
            detected_mask_pixels = np.count_nonzero([self.list_cell_masks.flatten(), self.list_nuc_masks.flatten(),
                                                     self.list_cyto_masks_no_nuclei.flatten()])
            self.number_detected_cells = np.max(self.list_cell_masks)
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
            self.number_detected_cells = np.max(self.list_cell_masks)
            print('    Number of detected cells:                ', self.number_detected_cells)
            if self.nucChannel is not None:
                mask_nuc_path = os.path.join(self.step_output_dir,
                                                                   'masksNuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_nuc_path, self.list_nuc_masks)
            if self.cytoChannel is not None:
                mask_cyto_path = os.path.join(self.step_output_dir,
                                                                    'masksCyto_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_path, self.list_cell_masks)
            if self.cytoChannel is not None and self.nucChannel is not None:
                mask_cyto_no_nuclei_path = os.path.join(self.step_output_dir,
                                                                              'masksCytoNoNuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_no_nuclei_path, self.list_cyto_masks_no_nuclei)

    def load_in_already_made_masks(self):
        # Paths to masks
        if not Utilities().is_None(self.nucChannel):
            mask_nuc_path = self.local_mask_folder.absolute().joinpath('masks_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.list_nuc_masks = imread(str(mask_nuc_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.list_nuc_masks)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing nuc masks')
        if not Utilities().is_None(self.cytoChannel):
            mask_cyto_path = self.local_mask_folder.absolute().joinpath('masks_cyto_' + self.temp_file_name + '.tif')
            try:
                self.list_cell_masks = imread(str(mask_cyto_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.list_cell_masks)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto masks.')
        if ((not Utilities().is_None(self.nucChannel)) and
                (not Utilities().is_None(self.cytoChannel))):
            self.mask_cyto_no_nuclei_path = self.local_mask_folder.absolute().joinpath(
                'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.list_cyto_masks_no_nuclei = imread(str(self.mask_cyto_no_nuclei_path))
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto only masks.')
        # test all masks exist, if not create the variable and set as None.
        if not 'list_nuc_masks' in locals():
            self.list_nuc_masks = None
        if not 'list_cell_masks' in locals():
            self.list_cell_masks = None
        if not 'list_cyto_masks_no_nuclei' in locals():
            self.list_cyto_masks_no_nuclei = None


class CellSegmentationStepClass(SequentialStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.save_masks_as_file = None
        self.nucChannel = None
        self.cytoChannel = None
        self.temp_file_name = None
        self.number_detected_cells = None
        self.segmentation_successful = None
        self.list_cyto_masks_no_nuclei = None
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
                self.list_cell_masks, self.list_nuc_masks, self.list_cyto_masks_no_nuclei = (
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
                CellSegmentationOutput(list_cell_masks=self.list_cell_masks,
                                       list_nuc_masks=self.list_nuc_masks,
                                       list_cyto_masks=self.list_cyto_masks_no_nuclei,
                                       segmentation_successful=segmentation_successful,
                                       number_detected_cells=self.number_detected_cells,
                                       id=id))

            return single_image_cell_segmentation_output

    def test_segementation(self):
        # test if segmentation was successful
        if self.cytoChannel is None:
            detected_mask_pixels = np.count_nonzero([self.list_nuc_masks.flatten()])
            self.number_detected_cells = np.max(self.list_nuc_masks)
        if self.nucChannel is None:
            detected_mask_pixels = np.count_nonzero([self.list_cell_masks.flatten()])
            self.number_detected_cells = np.max(self.list_cell_masks)
        if self.nucChannel is not None and self.cytoChannel is not None:
            detected_mask_pixels = np.count_nonzero([self.list_cell_masks.flatten(), self.list_nuc_masks.flatten(),
                                                     self.list_cyto_masks_no_nuclei.flatten()])
            self.number_detected_cells = np.max(self.list_cell_masks)
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
            self.number_detected_cells = np.max(self.list_cell_masks)
            print('    Number of detected cells:                ', self.number_detected_cells)
            if self.nucChannel is not None:
                mask_nuc_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                   'masks_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_nuc_path, self.list_nuc_masks)
            if self.cytoChannel is not None:
                mask_cyto_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                    'masks_cyto_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_path, self.list_cell_masks)
            if self.cytoChannel is not None and self.nucChannel is not None:
                mask_cyto_no_nuclei_path = pathlib.Path().absolute().joinpath(self.masks_folder_name,
                                                                              'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
                tifffile.imwrite(mask_cyto_no_nuclei_path, self.list_cyto_masks_no_nuclei)

    def load_in_already_made_masks(self):
        # Paths to masks
        if not Utilities().is_None(self.nucChannel):
            mask_nuc_path = self.local_mask_folder.absolute().joinpath('masks_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.list_nuc_masks = imread(str(mask_nuc_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.list_nuc_masks)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing nuc masks')
        if not Utilities().is_None(self.cytoChannel):
            mask_cyto_path = self.local_mask_folder.absolute().joinpath('masks_cyto_' + self.temp_file_name + '.tif')
            try:
                self.list_cell_masks = imread(str(mask_cyto_path))
                self.segmentation_successful = True
                self.number_detected_cells = np.max(self.list_cell_masks)
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto masks.')
        if ((not Utilities().is_None(self.nucChannel)) and
                (not Utilities().is_None(self.cytoChannel))):
            self.mask_cyto_no_nuclei_path = self.local_mask_folder.absolute().joinpath(
                'masks_cyto_no_nuclei_' + self.temp_file_name + '.tif')
            try:
                self.list_cyto_masks_no_nuclei = imread(str(self.mask_cyto_no_nuclei_path))
            except:
                self.segmentation_successful = False
                print('    Segmentation was not successful. Due to missing cyto only masks.')
        # test all masks exist, if not create the variable and set as None.
        if not 'list_nuc_masks' in locals():
            self.list_nuc_masks = None
        if not 'list_cell_masks' in locals():
            self.list_cell_masks = None
        if not 'list_cyto_masks_no_nuclei' in locals():
            self.list_cyto_masks_no_nuclei = None


class BIGFISH_Tensorflow_Segmentation(SequentialStepsClass):
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
        # output = CellSegmentationOutput(list_cell_masks=cell_label, list_nuc_masks=nuc_label,
        #                                 list_cyto_masks=mask_cytosol, segmentation_successful=1,
        #                                 number_detected_cells=num_of_cells, id=id)
        # return output