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

    def load_in_attributes(self):
        # INPUTS
        self.name_for_files = self.experiment.initial_data_location.name
        self.list_images = self.pipelineData.list_images[:min(self.pipelineSettings.user_select_number_of_images_to_run,
                                                              len(self.pipelineData.list_images))]
        self.list_files_names = self.pipelineData.list_image_names[
                                :min(self.pipelineSettings.user_select_number_of_images_to_run,
                                     len(self.pipelineData.list_images))]
        self.list_segmentation_successful = self.pipelineData.pipelineOutputs.CellSegmentationOutput.segmentation_successful[
                                            :min(self.pipelineSettings.user_select_number_of_images_to_run,
                                                 len(self.pipelineData.list_images))]
        self.list_is_image_sharp = self.pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_is_image_sharp[
                                   :min(self.pipelineSettings.user_select_number_of_images_to_run,
                                        len(self.pipelineData.list_images))]
        self.list_processing_successful = [a and b for a, b in zip(self.list_segmentation_successful, self.list_is_image_sharp)]
        self.save_files = self.pipelineSettings.save_files
        self.show_plots = self.pipelineSettings.show_plots
        self.temp_folder_name = self.pipelineData.temp_folder_name
        self.dataframe = self.pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.dfFISH
        self.channels_with_FISH = self.experiment.FISHChannel
        self.channels_with_cytosol = self.experiment.cytoChannel
        self.channels_with_nucleus = self.experiment.nucChannel
        self.list_masks_complete_cells = self.pipelineData.pipelineOutputs.CellSegmentationOutput.masks_complete_cells
        self.list_masks_nuclei = self.pipelineData.pipelineOutputs.CellSegmentationOutput.masks_nuclei
        self.data_folder_path = self.experiment.initial_data_location
        self.diameter_cytosol = self.pipelineSettings.diameter_cytosol
        self.diameter_nucleus = self.pipelineSettings.diameter_nucleus
        self.minimum_spots_cluster = self.pipelineSettings.minimum_spots_cluster
        self.CLUSTER_RADIUS = self.pipelineSettings.CLUSTER_RADIUS
        self.list_voxels = []
        self.list_psfs = []
        for i in range(len(self.channels_with_FISH)):
            self.list_voxels.append([self.experiment.voxel_size_z, self.terminatorScope.voxel_size_yx])
            self.list_psfs.append([self.terminatorScope.psf_z, self.terminatorScope.psf_yx])
        self.threshold_for_spot_detection = self.pipelineData.prePipelineOutputs.AutomaticThresholdingOutput.threshold_for_spot_detection
        self.list_average_spots_per_cell = self.pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.avg_number_of_spots_per_cell_each_ch
        self.list_number_detected_cells = self.pipelineData.pipelineOutputs.CellSegmentationOutput.number_detected_cells
        self.list_metric_sharpness_images = self.pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_metric_sharpness_images
        self.remove_out_of_focus_images = self.pipelineSettings.remove_out_of_focus_images
        self.sharpness_threshold = self.pipelineSettings.sharpness_threshold
        self.remove_z_slices_borders = self.pipelineSettings.remove_z_slices_borders
        self.NUMBER_Z_SLICES_TO_TRIM = self.pipelineSettings.NUMBER_Z_SLICES_TO_TRIM
        self.list_counter_image_id = self.pipelineData.pipelineOutputs.CellSegmentationOutput.img_id
        self.number_of_images_to_process = self.experiment.number_of_images_to_process
        self.list_thresholds_spot_detection = self.pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.thresholds_spot_detection
        self.save_pdf_report = self.pipelineSettings.save_pdf_report
        self.list_z_slices_per_image = self.pipelineData.prePipelineOutputs.TrimZSlicesOutput.list_z_slices_per_image
        self.save_all_images = self.pipelineSettings.save_all_images

    def main(self):
        # Saving all original images as a PDF
        image_name = 'original_images_' + self.name_for_files + '.pdf'
        if self.save_files:
            Plots().plotting_all_original_images(self.list_images, self.list_files_names, image_name,
                                                 show_plots=self.show_plots)

        # Creating an image with all segmentation results
        image_name = 'segmentation_images_' + self.name_for_files + '.pdf'
        if self.save_files:
            Plots().plotting_segmentation_images(directory=pathlib.Path().absolute().joinpath(self.temp_folder_name),
                                                 list_files_names=self.list_files_names,
                                                 list_segmentation_successful=self.list_processing_successful,
                                                 image_name=image_name,
                                                 show_plots=False)

        # Saving all cells in a single image file
        if self.save_files:
            for k in range(len(self.channels_with_FISH)):
                Plots().plot_all_cells_and_spots(list_images=self.list_images,
                                                 complete_dataframe=self.dataframe,
                                                 selected_channel=self.channels_with_FISH[k],
                                                 list_masks_complete_cells=self.list_masks_complete_cells,
                                                 list_masks_nuclei=self.list_masks_nuclei,
                                                 spot_type=k,
                                                 list_segmentation_successful=self.list_processing_successful,
                                                 image_name='cells_channel_' + str(
                                                     self.channels_with_FISH[k]) + '_' + self.name_for_files + '.pdf',
                                                 microns_per_pixel=None,
                                                 show_legend=True,
                                                 show_plot=False)
        # Creating the dataframe    
        if self.save_files:
            if (not str(self.name_for_files)[0:5] == 'temp_') and np.sum(self.list_processing_successful) > 0:
                self.dataframe.to_csv('dataframe_' + self.name_for_files + '.csv')
            elif np.sum(self.list_processing_successful) > 0:
                self.dataframe.to_csv('dataframe_' + self.name_for_files[5:] + '.csv')

        # Creating the metadata
        if self.save_files:
            Metadata(self.data_folder_path,
                     self.channels_with_cytosol,
                     self.channels_with_nucleus,
                     self.channels_with_FISH,
                     self.diameter_nucleus,
                     self.diameter_cytosol,
                     self.minimum_spots_cluster,
                     list_voxels=self.list_voxels,
                     list_psfs=self.list_psfs,
                     file_name_str=self.name_for_files,
                     list_segmentation_successful=self.list_segmentation_successful,
                     list_counter_image_id=self.list_counter_image_id,
                     threshold_for_spot_detection=self.threshold_for_spot_detection,
                     number_of_images_to_process=self.number_of_images_to_process,
                     remove_z_slices_borders=self.remove_z_slices_borders,
                     NUMBER_Z_SLICES_TO_TRIM=self.NUMBER_Z_SLICES_TO_TRIM,
                     CLUSTER_RADIUS=self.CLUSTER_RADIUS,
                     list_thresholds_spot_detection=self.list_thresholds_spot_detection,
                     list_average_spots_per_cell=self.list_average_spots_per_cell,
                     list_number_detected_cells=self.list_number_detected_cells,
                     list_is_image_sharp=self.list_is_image_sharp,
                     list_metric_sharpeness_images=self.list_metric_sharpness_images,
                     remove_out_of_focus_images=self.remove_out_of_focus_images,
                     sharpness_threshold=self.sharpness_threshold).write_metadata()

        # Creating a PDF report
        if self.save_pdf_report and self.save_files:
            filenames_for_pdf_report = [f[:-4] for f in self.list_files_names]
            ReportPDF(directory=pathlib.Path().absolute().joinpath(self.temp_folder_name),
                      filenames_for_pdf_report=filenames_for_pdf_report,
                      channels_with_FISH=self.channels_with_FISH,
                      save_all_images=self.save_all_images,
                      list_z_slices_per_image=self.list_z_slices_per_image,
                      threshold_for_spot_detection=self.threshold_for_spot_detection,
                      list_segmentation_successful=self.list_processing_successful).create_report()


class AddTPs2FISHDataFrame(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self):
        pass
