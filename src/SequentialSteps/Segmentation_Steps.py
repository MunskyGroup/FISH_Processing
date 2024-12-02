import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
import bigfish
import bigfish.stack as stack
# import bigfish.segmentation as segmentation
import bigfish.multistack as multistack
import bigfish.plot as plot
import dask.array as da
from abc import ABC, abstractmethod



from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src import SequentialStepsClass
from src.GeneralOutput import OutputClass
from src.Parameters import Parameters


#%% Abstract Class
class CellSegmentationOutput(OutputClass):
    def append(self, timepoint, position, nuc_mask, cell_mask, nucChannel, cytoChannel, masks):
        if timepoint == 0:
            if nuc_mask is not None and len(nuc_mask.shape) != 2:
                raise ValueError('Nuclei mask must be 2D')
            
            if cell_mask is not None and len(cell_mask.shape) != 2:
                raise ValueError('Cell mask must be 2D')
            
            if nuc_mask is not None:
                masks[position, 0, nucChannel, 0, :, :] = nuc_mask
            
            if cell_mask is not None:
                masks[position, 0, cytoChannel, 0, :, :] = cell_mask

            Parameters.update_parameters(kwargs={'masks': masks})

class CellSegmentation(SequentialStepsClass):
    def main(self, masks, image, fov, timepoint, nucChannel, cytoChannel, 
             display_plots, do_3D_Segmentation, **kwargs):
        if len(image.shape) == 3:
            image = np.max(image, axis=0) 
        
        if timepoint == 0:
            nuc_mask = self.segment_nuclei(**kwargs)

            cell_mask = self.segment_cells(**kwargs)

            nuc_mask, cell_mask = self.align_nuc_cell_masks(nuc_mask, cell_mask, **kwargs)

            self.plot_segmentation(display_plots, image, nuc_mask, cell_mask, nucChannel, cytoChannel, do_3D_Segmentation)

            CellSegmentationOutput(timepoint, fov, nuc_mask, cell_mask, nucChannel, cytoChannel, masks)

    @abstractmethod
    def segment_nuclei(self, **kwargs):
        pass

    @abstractmethod
    def segment_cells(self, **kwargs):
        pass

    def align_nuc_cell_masks(self, nuc_mask, cell_mask):
        if nuc_mask is not None and cell_mask is not None:
            nuc_mask, cell_mask = multistack.match_nuc_cell(nuc_mask, cell_mask, single_nuc=False, cell_alone=False)
        return nuc_mask, cell_mask

    def plot_segmentation(self, display_plots, image, nuc_mask, cell_mask, nuc_channel, cyto_channel, do_3D_Segmentation):
        if display_plots:
            num_sub_plots = 0
            if nuc_mask is not None:
                num_sub_plots += 2
            if cell_mask is not None:
                num_sub_plots += 2
            fig, axs = plt.subplots(1, num_sub_plots, figsize=(12, 5))
            i = 0
            if nuc_mask is not None:
                if do_3D_Segmentation:
                    axs[i].imshow(np.max(image,axis=0)[nuc_channel, :, :])
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(np.max(nuc_mask,axis=0))
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[nuc_channel,:,:])
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(nuc_mask)
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
            if cell_mask is not None:
                if do_3D_Segmentation:
                    axs[i].imshow(np.max(image,axis=0)[cyto_channel, :, :])
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(np.max(cell_mask, axis=0))
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[cyto_channel,:,:])
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(cell_mask)
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
            plt.tight_layout()
            plt.show()

    def first_run(self, params):
        if params['masks'] is None:
            images = params['images']
            # TODO: make this smarter, we may want time, and z specific masks
            # add a flag so that these that t and z are not 1
            params['masks'] = da.zeros((images.shape[0], 1, images.shape[2], 1, images.shape[4], images.shape[5]), dtype=np.uint8)



    
#%% Useful functions
def remove_lonely_cells(cellmask, nucmask):
    # check to make sure there is only one nuc label per cell label
    # if not, remove both the cell and nuc labels
    for label in np.unique(cellmask):
        if label == 0:
            continue
        pixels = nucmask[cellmask == label].flatten()
        # remove zero pixels
        pixels = pixels[pixels != 0]
        if len(np.unique(pixels)) > 2:
            print('Removing cell with multiple nuclei, label:', label)
            nucmask[cellmask == label] = 0
            cellmask[cellmask == np.unique(pixels)] = 0
            
    for label in np.unique(nucmask):
        if label == 0:
            continue
        pixels = cellmask[nucmask == label].flatten()
        if len(np.unique(pixels)) > 2:
            print('Removing nucleus with multiple cells, label:', label)
            nucmask[nucmask == label] = 0
            cellmask[nucmask == label] = 0
    return cellmask, nucmask

def confirm_labels(nucmask, cellmask):
    # relabels all the cells masks so they agree with the nuclei masks
    # this is done by checking the nuclei mask for each cell and assigning the cell the label of the nucleus
    bad_labels = [] # list of [nuc_label, cell_label]
    for label in np.unique(nucmask):
        if label == 0:
            continue
        nuc_label = np.argmax(np.bincount(nucmask[cellmask == label].flatten()))
        cellmask[cellmask == label] = nuc_label





#%% Steps
class DilationedCytoMask(CellSegmentation):
    def __init__(self):
        super().__init__()

    def main(self, id, cyto_mask, nuc_mask, dilation_size: int = 5, **kwargs):
        if cyto_mask is None:
            cell_mask = np.zeros_like(nuc_mask)
            for label in np.unique(nuc_mask):
                if label == 0:
                    continue
                dilated_mask = np.zeros_like(nuc_mask)
                mask = nuc_mask == label
                dilated_mask += sk.morphology.binary_dilation(mask, selem=sk.morphology.disk(dilation_size))
                cell_mask[dilated_mask] = label
        
        cyto_mask = cell_mask.copy()
        cyto_mask[nuc_mask > 0] = 0

        return CellSegmentationOutput(list_cell_masks=cell_mask, list_cyto_masks=cyto_mask)


class SimpleCellposeSegmentaion(CellSegmentation):
    """
    A class for performing cell segmentation using the Cellpose model.
    Methods
    -------
    main(image, cytoChannel, nucChannel, masks, timepoint, fov, cellpose_model_type, cellpose_diameter, 
         cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation, cellpose_min_size, 
         cellpose_flow_threshold, cellpose_cellprob_threshold, cellpose_pretrained_model, display_plots, **kwargs)
        Main method to perform segmentation on the given image.
    Parameters
    ----------
    image : ndarray
        The input image to be segmented.
    cytoChannel : int
        The channel index for cytoplasm.
    nucChannel : int
        The channel index for nuclei.
    masks : list
        List to store the segmentation masks.
    timepoint : int
        The timepoint index for the image.
    fov : int
        The field of view index for the image.
    cellpose_model_type : str or list of str, optional
        The type of Cellpose model to use. Default is ['cyto3', 'nuclei'].
    cellpose_diameter : float or list of float, optional
        The diameter of the cells to be segmented. Default is 180.
    cellpose_channel_axis : int, optional
        The axis of the channels in the image. Default is 0.
    cellpose_invert : bool or list of bool, optional
        Whether to invert the image for segmentation. Default is False.
    cellpose_normalize : bool, optional
        Whether to normalize the image for segmentation. Default is True.
    do_3D_Segmentation : bool, optional
        Whether to perform 3D segmentation. Default is False.
    cellpose_min_size : float or list of float, optional
        The minimum size of the cells to be segmented. Default is 500.
    cellpose_flow_threshold : float or list of float, optional
        The flow threshold for the Cellpose model. Default is 0.
    cellpose_cellprob_threshold : float or list of float, optional
        The cell probability threshold for the Cellpose model. Default is 0.
    cellpose_pretrained_model : str or list of str, optional
        The path to the pretrained Cellpose model. Default is False.
    display_plots : bool, optional
        Whether to display plots of the segmentation results. Default is False.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self):
        super().__init__()

    def main(self, 
             image, 
             cytoChannel, 
             nucChannel, 
             masks,
             timepoint: int,
             fov: int,
             cellpose_model_type: str | list[str] = ['cyto3', 'nuclei'], 
             cellpose_diameter: float | list[float] = 180, 
             cellpose_channel_axis: int = 0,
             cellpose_invert: bool | list = False, 
             cellpose_normalize: bool = True, 
             do_3D_Segmentation: bool = False, # This is not implemented
             cellpose_min_size: float | list[float] = 500, 
             cellpose_flow_threshold: float | list[float] = 0, 
             cellpose_cellprob_threshold: float | list[float] = 0,
             cellpose_pretrained_model: str | list[str] = False,
             display_plots: bool = False,
               **kwargs):
        if image.shape[1] >= 1:
            image = np.max(image, axis=1) 
        
        if timepoint == 0:
            nuc_mask = self.segment_nuclei(image, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                                           cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                                           cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation,
                                           cellpose_pretrained_model)

            cell_mask = self.segment_cells(image, cytoChannel, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                                           cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                                           cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation, 
                                           cellpose_pretrained_model)

            nuc_mask, cell_mask = self.align_nuc_cell_masks(nuc_mask, cell_mask)

            self.plot_segmentation(display_plots, image, nuc_mask, cell_mask, nucChannel, cytoChannel, do_3D_Segmentation)

            CellSegmentationOutput(timepoint, fov, nuc_mask, cell_mask, nucChannel, cytoChannel, masks)
        
    def unpack_lists(self, cellpose_min_size, 
             cellpose_flow_threshold, 
             cellpose_cellprob_threshold,
             cellpose_model_type,
             cellpose_diameter,
             pretrained_model):
        if isinstance(cellpose_min_size, list):
            nuc_min_size = cellpose_min_size[1]
            cyto_min_size = cellpose_min_size[0]
        else:
            nuc_min_size = cellpose_min_size
            cyto_min_size = cellpose_min_size
        
        if isinstance(cellpose_flow_threshold, list):
            nuc_flow_threshold = cellpose_flow_threshold[1]
            cyto_flow_threshold = cellpose_flow_threshold[0]
        else:
            nuc_flow_threshold = cellpose_flow_threshold
            cyto_flow_threshold = cellpose_flow_threshold
        
        if isinstance(cellpose_cellprob_threshold, list):
            nuc_cellprob_threshold = cellpose_cellprob_threshold[1]
            cyto_cellprob_threshold = cellpose_cellprob_threshold[0]
        else:
            nuc_cellprob_threshold = cellpose_cellprob_threshold
            cyto_cellprob_threshold = cellpose_cellprob_threshold

        if isinstance(cellpose_model_type, list):
            nuc_model_type = cellpose_model_type[1]
            cyto_model_type = cellpose_model_type[0]
        else:
            nuc_model_type = cellpose_model_type
            cyto_model_type = cellpose_model_type

        if isinstance(cellpose_diameter, list):
            nuc_diameter = cellpose_diameter[1]
            cyto_diameter = cellpose_diameter[0]
        else:
            nuc_diameter = cellpose_diameter
            cyto_diameter = cellpose_diameter

        if isinstance(pretrained_model, list):
            nuc_pretrained_model = pretrained_model[1]
            cyto_pretrained_model = pretrained_model[0]
        else:
            nuc_pretrained_model = pretrained_model
            cyto_pretrained_model = pretrained_model

        return (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, nuc_cellprob_threshold, 
                cyto_cellprob_threshold, nuc_model_type, cyto_model_type, nuc_diameter, cyto_diameter, nuc_pretrained_model,
                cyto_pretrained_model) 

    def segment_nuclei(self, image, nucChannel, cellpose_min_size, cellpose_flow_threshold, 
                       cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter, 
                       cellpose_channel_axis, cellpose_invert, cellpose_normalize, cellpose_do_3D,
                       cellpose_pretrained_model):
        if nucChannel is not None:
            (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, 
            nuc_cellprob_threshold, cyto_cellprob_threshold, nuc_model_type, 
            cyto_model_type, nuc_diameter, cyto_diameter, 
            nuc_pretrained_model, cyto_pretrained_model) = self.unpack_lists(cellpose_min_size, 
                                                                        cellpose_flow_threshold, 
                                                                        cellpose_cellprob_threshold,
                                                                        cellpose_model_type,
                                                                        cellpose_diameter,
                                                                        cellpose_pretrained_model)
            model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            nuc_pretrained_model = os.path.join(model_location, nuc_pretrained_model) if nuc_pretrained_model else None

            cp = models.CellposeModel(model_type=nuc_model_type, gpu=True, pretrained_model=nuc_pretrained_model)
            sz = models.SizeModel(cp)

            model = models.Cellpose(gpu=True)
            model.cp = cp
            model.sz = sz
            # nucmodel = models.Cellpose(model_type=nuc_model_type, gpu=True)
            # if cp is not None:
            #     nucmodel.cp = cp
            channels = [0, 0]
            nuc_image = image[nucChannel, :, :].compute()
            nuc_mask, flows, styles, diams = model.eval(nuc_image,
                                                channels=channels, 
                                                diameter=nuc_diameter, 
                                                invert=cellpose_invert, 
                                                normalize=cellpose_normalize, 
                                                channel_axis=cellpose_channel_axis, 
                                                do_3D=cellpose_do_3D,
                                                min_size=nuc_min_size, 
                                                flow_threshold=nuc_flow_threshold, 
                                                cellprob_threshold=nuc_cellprob_threshold)
                                                # net_avg=True


            return nuc_mask
        
    def segment_cells(self, image, cytoChannel, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                      cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                      cellpose_channel_axis, cellpose_invert, cellpose_normalize, cellpose_do_3D,
                      cellpose_pretrained_model):
        if cytoChannel is not None:
            (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, 
            nuc_cellprob_threshold, cyto_cellprob_threshold, nuc_model_type,
            cyto_model_type, nuc_diameter, cyto_diameter, 
            nuc_pretrained_model, cyto_pretrained_model) = self.unpack_lists(cellpose_min_size, 
                                                                        cellpose_flow_threshold, 
                                                                        cellpose_cellprob_threshold,
                                                                        cellpose_model_type,
                                                                        cellpose_diameter,
                                                                        cellpose_pretrained_model)
            
            model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            cyto_pretrained_model = os.path.join(model_location, cyto_pretrained_model) if cyto_pretrained_model else None


            cp = models.CellposeModel(model_type=cyto_model_type, gpu=True, pretrained_model=cyto_pretrained_model)
            sz = models.SizeModel(cp)

            model = models.Cellpose(gpu=True)
            model.cp = cp
            model.sz = sz

            channels = [0, 0]
            cyto_image = image[cytoChannel, :, :].compute()
            cell_mask, flows, styles, diams = model.eval(cyto_image,
                                                    channels=channels, 
                                                    diameter=cyto_diameter, 
                                                    invert=cellpose_invert, 
                                                    normalize=cellpose_normalize, 
                                                    channel_axis=cellpose_channel_axis, 
                                                    do_3D=cellpose_do_3D,
                                                    min_size=cyto_min_size, 
                                                    flow_threshold=cyto_flow_threshold, 
                                                    cellprob_threshold=cyto_cellprob_threshold,)
            return cell_mask

class CellSegmentationStepClass_JF(CellSegmentation):
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


class CellSegmentationStepClass_Luis(SequentialStepsClass):
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




if __name__ == '__main__':
    pass
    # print(os.path.dirname(os.path.dirname(__file__)))
