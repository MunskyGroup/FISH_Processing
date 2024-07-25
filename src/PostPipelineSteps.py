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

    def main(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, experiment:Experiment, terminatorScope:ScopeClass):
        # INPUTS
        name_for_files = experiment.initial_data_location.name
        list_images = pipelineData.list_images[:min(pipelineSettings.user_select_number_of_images_to_run, len(pipelineData.list_images))]
        list_files_names = pipelineData.list_image_names[:min(pipelineSettings.user_select_number_of_images_to_run, len(pipelineData.list_images))] 
        list_segmentation_successful = pipelineData.pipelineOutputs.CellSegmentationOutput.segmentation_successful[:min(pipelineSettings.user_select_number_of_images_to_run, len(pipelineData.list_images))]
        list_is_image_sharp = pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_is_image_sharp[:min(pipelineSettings.user_select_number_of_images_to_run, len(pipelineData.list_images))]
        list_processing_successful = [a and b for a, b in zip(list_segmentation_successful, list_is_image_sharp)]
        save_files = pipelineSettings.save_files
        show_plots = pipelineSettings.show_plots
        temp_folder_name = pipelineData.temp_folder_name
        dataframe = pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.dfFISH
        channels_with_FISH = experiment.FISHChannel
        channels_with_cytosol = experiment.cytoChannel
        channels_with_nucleus = experiment.nucChannel
        list_masks_complete_cells = pipelineData.pipelineOutputs.CellSegmentationOutput.masks_complete_cells
        list_masks_nuclei = pipelineData.pipelineOutputs.CellSegmentationOutput.masks_nuclei
        data_folder_path = experiment.initial_data_location
        diameter_cytosol = pipelineSettings.diameter_cytosol
        diameter_nucleus = pipelineSettings.diameter_nucleus
        minimum_spots_cluster = pipelineSettings.minimum_spots_cluster
        CLUSTER_RADIUS = pipelineSettings.CLUSTER_RADIUS
        list_voxels = []
        list_psfs = []
        for i in range (len(channels_with_FISH)):
            list_voxels.append([experiment.voxel_size_z,terminatorScope.voxel_size_yx])
            list_psfs.append([terminatorScope.psf_z, terminatorScope.psf_yx])
        threshold_for_spot_detection = pipelineData.prePipelineOutputs.AutomaticThresholdingOutput.threshold_for_spot_detection
        list_average_spots_per_cell = pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.avg_number_of_spots_per_cell_each_ch
        list_number_detected_cells = pipelineData.pipelineOutputs.CellSegmentationOutput.number_detected_cells
        list_metric_sharpeness_images = pipelineData.prePipelineOutputs.CalculateSharpnessOutput.list_metric_sharpeness_images
        remove_out_of_focus_images = pipelineSettings.remove_out_of_focus_images
        sharpness_threshold = pipelineSettings.sharpness_threshold
        remove_z_slices_borders = pipelineSettings.remove_z_slices_borders
        NUMBER_Z_SLICES_TO_TRIM = pipelineSettings.NUMBER_Z_SLICES_TO_TRIM
        list_counter_image_id = pipelineData.pipelineOutputs.CellSegmentationOutput.img_id
        number_of_images_to_process = experiment.number_of_images_to_process
        list_thresholds_spot_detection = pipelineData.pipelineOutputs.SpotDetectionStepOutputClass.thresholds_spot_detection
        save_pdf_report = pipelineSettings.save_pdf_report
        list_z_slices_per_image = pipelineData.prePipelineOutputs.TrimZSlicesOutput.list_z_slices_per_image
        save_all_images = pipelineSettings.save_all_images


        # LUIS'S CODE
        # Saving all original images as a PDF
        #print('- CREATING THE PLOT WITH ORIGINAL IMAGES')
        image_name= 'original_images_' + name_for_files +'.pdf'
        if save_files == True:
            Plots().plotting_all_original_images(list_images,list_files_names,image_name,show_plots=show_plots)
        # Creating an image with all segmentation results
        image_name= 'segmentation_images_' + name_for_files +'.pdf'
        if save_files == True:
            Plots().plotting_segmentation_images(directory=pathlib.Path().absolute().joinpath(temp_folder_name),
                                            list_files_names=list_files_names,
                                            list_segmentation_successful=list_processing_successful,
                                            image_name=image_name,
                                            show_plots=False)
        # Saving all cells in a single image file
        #print('- CREATING THE PLOT WITH ALL CELL IMAGES') 
        if save_files == True:
            for k in range (len(channels_with_FISH)):
                Plots().plot_all_cells_and_spots(list_images=list_images, 
                                            complete_dataframe=dataframe, 
                                            selected_channel=channels_with_FISH[k], 
                                            list_masks_complete_cells = list_masks_complete_cells,
                                            list_masks_nuclei = list_masks_nuclei,
                                            spot_type=k,
                                            list_segmentation_successful=list_processing_successful,
                                            image_name='cells_channel_'+ str(channels_with_FISH[k])+'_'+ name_for_files +'.pdf',
                                            microns_per_pixel=None,
                                            show_legend = True,
                                            show_plot= False)
        # Creating the dataframe    
        if save_files == True:   
            if  (not str(name_for_files)[0:5] ==  'temp_') and np.sum(list_processing_successful)>0:
                dataframe.to_csv('dataframe_' + name_for_files +'.csv')
            elif np.sum(list_processing_successful)>0:
                dataframe.to_csv('dataframe_' + name_for_files[5:] +'.csv')        
        # Creating the metadata
        #print('- CREATING THE METADATA FILE')
        if save_files == True:
            Metadata(data_folder_path, 
                channels_with_cytosol, 
                channels_with_nucleus, 
                channels_with_FISH,
                diameter_nucleus, 
                diameter_cytosol, 
                minimum_spots_cluster,
                list_voxels=list_voxels, 
                list_psfs=list_psfs,
                file_name_str=name_for_files,
                list_segmentation_successful=list_segmentation_successful,
                list_counter_image_id=list_counter_image_id,
                threshold_for_spot_detection=threshold_for_spot_detection,
                number_of_images_to_process=number_of_images_to_process,
                remove_z_slices_borders=remove_z_slices_borders,
                NUMBER_Z_SLICES_TO_TRIM=NUMBER_Z_SLICES_TO_TRIM,
                CLUSTER_RADIUS=CLUSTER_RADIUS,
                list_thresholds_spot_detection=list_thresholds_spot_detection,
                list_average_spots_per_cell=list_average_spots_per_cell,
                list_number_detected_cells=list_number_detected_cells,
                list_is_image_sharp=list_is_image_sharp,
                list_metric_sharpeness_images=list_metric_sharpeness_images,
                remove_out_of_focus_images=remove_out_of_focus_images,
                sharpness_threshold=sharpness_threshold).write_metadata()
        # Creating a PDF report
        #print('CREATING THE PDF REPORT')
        if (save_pdf_report ==True) and (save_files == True):
            filenames_for_pdf_report = [ f[:-4] for f in list_files_names]
            ReportPDF(directory=pathlib.Path().absolute().joinpath(temp_folder_name), 
                    filenames_for_pdf_report=filenames_for_pdf_report, 
                    channels_with_FISH=channels_with_FISH, 
                    save_all_images=save_all_images, 
                    list_z_slices_per_image=list_z_slices_per_image,
                    threshold_for_spot_detection=threshold_for_spot_detection,
                    list_segmentation_successful=list_processing_successful ).create_report()





class AddTPs2FISHDataFrame(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, experiment:Experiment, terminatorScope:ScopeClass):
        pass


