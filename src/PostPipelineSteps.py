import pathlib
import numpy as np

from src import PipelineSettings, Experiment, ScopeClass, PipelineDataClass, postPipelineStepsClass
from src.Util.Plots import Plots
from src.Util.Metadata import Metadata
from src.Util.ReportPDF import ReportPDF


# from GeneralStepClasses import postPipelineStepsClass
# from MicroscopeClass import ScopeClass
# from ExperimentClass import Experiment
# from PipelineSettings import PipelineSettings
# from PipelineDataClass import PipelineDataClass
# from Util import Plots, Metadata, ReportPDF

class SavePDFReport(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, initial_data_location, num_img_2_run, list_images,
        list_image_names, segmentation_successful, list_is_image_sharp, save_files, show_plots,
        temp_folder_name, dfFISH, FISHChannel, cytoChannel, nucChannel, masks_complete_cells, masks_nuclei,
        diameter_cytosol, diameter_nucleus, minimum_spots_cluster, CLUSTER_RADIUS, voxel_size_z,
        voxel_size_yx, psf_yx, psf_z, threshold_for_spot_detection, avg_number_of_spots_per_cell_each_ch,
        number_detected_cells,
        list_metric_sharpness_images,
        remove_out_of_focus_images, sharpness_threshold, remove_z_slices_borders,NUMBER_Z_SLICES_TO_TRIM, img_id,
        number_of_images_to_process, automatic_spot_detection_threshold, individual_threshold_spot_detection, save_pdf_report, list_z_slices_per_image,
        save_all_images, **kwargs):

        name_for_files = initial_data_location.name
        list_images = list_images[:num_img_2_run]
        list_files_names = list_image_names[:num_img_2_run]
        list_segmentation_successful = segmentation_successful[:num_img_2_run]
        list_is_image_sharp = list_is_image_sharp[:num_img_2_run]

        list_voxels = []
        list_psfs = []
        for i in range(len(FISHChannel)):
            list_voxels.append([voxel_size_z, voxel_size_yx])
            list_psfs.append([psf_z, psf_yx])

        list_processing_successful = [a and b for a, b in zip(list_segmentation_successful, list_is_image_sharp)]

        dataframe = dfFISH

        # Saving all original images as a PDF
        image_name = 'original_images_' + name_for_files + '.pdf'
        if save_files:
            Plots().plotting_all_original_images(list_images, list_files_names, image_name,
                                                 show_plots=show_plots)

        # Creating an image with all segmentation results
        image_name = 'segmentation_images_' + name_for_files + '.pdf'
        if save_files:
            Plots().plotting_segmentation_images(directory=pathlib.Path().absolute().joinpath(temp_folder_name),
                                                 list_files_names=list_files_names,
                                                 list_segmentation_successful=list_processing_successful,
                                                 image_name=image_name,
                                                 show_plots=False)

        # Saving all cells in a single image file
        if save_files:
            for k in range(len(FISHChannel)):
                Plots().plot_all_cells_and_spots(list_images=list_images,
                                                 complete_dataframe=dataframe,
                                                 selected_channel=FISHChannel[k],
                                                 list_masks_complete_cells=masks_complete_cells,
                                                 list_masks_nuclei=masks_nuclei,
                                                 spot_type=k,
                                                 list_segmentation_successful=list_processing_successful,
                                                 image_name='cells_channel_' + str(
                                                     FISHChannel[k]) + '_' + name_for_files + '.pdf',
                                                 microns_per_pixel=None,
                                                 show_legend=True,
                                                 show_plot=False)
        # Creating the dataframe    
        if save_files:
            if (not str(name_for_files)[0:5] == 'temp_') and np.sum(list_processing_successful) > 0:
                dataframe.to_csv('dataframe_' + name_for_files + '.csv')
            elif np.sum(list_processing_successful) > 0:
                dataframe.to_csv('dataframe_' + name_for_files[5:] + '.csv')

        # Creating the metadata
        if save_files:
            Metadata(initial_data_location,
                     cytoChannel,
                     nucChannel,
                     FISHChannel,
                     diameter_nucleus,
                     diameter_cytosol,
                     minimum_spots_cluster,
                     list_voxels=list_voxels,
                     list_psfs=list_psfs,
                     file_name_str=name_for_files,
                     list_segmentation_successful=list_segmentation_successful,
                     list_counter_image_id=img_id,
                     threshold_for_spot_detection=automatic_spot_detection_threshold,
                     number_of_images_to_process=number_of_images_to_process,
                     remove_z_slices_borders=remove_z_slices_borders,
                     NUMBER_Z_SLICES_TO_TRIM=NUMBER_Z_SLICES_TO_TRIM,
                     CLUSTER_RADIUS=CLUSTER_RADIUS,
                     list_thresholds_spot_detection=individual_threshold_spot_detection,
                     list_average_spots_per_cell=avg_number_of_spots_per_cell_each_ch,
                     list_number_detected_cells=number_detected_cells,
                     list_is_image_sharp=list_is_image_sharp,
                     list_metric_sharpeness_images=list_metric_sharpness_images,
                     remove_out_of_focus_images=remove_out_of_focus_images,
                     sharpness_threshold=sharpness_threshold).write_metadata()

        # Creating a PDF report
        if save_pdf_report and save_files:
            filenames_for_pdf_report = [f[:-4] for f in list_files_names]
            ReportPDF(directory=pathlib.Path().absolute().joinpath(temp_folder_name),
                      filenames_for_pdf_report=filenames_for_pdf_report,
                      channels_with_FISH=FISHChannel,
                      save_all_images=save_all_images,
                      list_z_slices_per_image=list_z_slices_per_image,
                      threshold_for_spot_detection=threshold_for_spot_detection,
                      list_segmentation_successful=list_processing_successful).create_report()


class AddTPs2FISHDataFrame(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self):
        pass
