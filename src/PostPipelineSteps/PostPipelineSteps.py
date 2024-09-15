import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd

from src import PipelineSettings, Experiment, ScopeClass, PipelineDataClass, postPipelineStepsClass
from src.Util.Plots import Plots
from src.Util.Metadata import Metadata
from src.Util.ReportPDF import ReportPDF
from src.Util.Utilities import Utilities
from src.Util.NASConnection import NASConnection



# from GeneralStepClasses import postPipelineStepsClass
# from MicroscopeClass import ScopeClass
# from ExperimentClass import Experiment
# from PipelineSettings import PipelineSettings
# from PipelineDataClass import PipelineDataClass
# from Util import Plots, Metadata, ReportPDF

#%% Jack
class BuildPDFReport(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, output_location, analysis_location, **kwargs):
        
        self.pdf = FPDF()
        WIDTH = 210
        HEIGHT = 297
        self.pdf.set_font('Arial', 'B', 14)

        steps = os.listdir(output_location)

        for step in steps:
            self.pdf.add_page()
            files = os.listdir(os.path.join(output_location,step))
            if len(files) == 0:
                break
            self.pdf.cell(w=0, h=0, txt=f'-------------------------{step}-------------------------', ln=2, align='C')
            file_types = []
            img_names = []
            for file in files:
                file_type = file.split('_')[0]
                img_names.append('_'.join(file.split('_')[1:]))
                file_types.append(file_type)
            file_types = list(set(file_types))
            img_names = list(set(img_names))

            for img_name in img_names:
                for file_type in file_types:
                    for file in files:
                        if img_name in file and file_type in file:
                            self.pdf.cell(w=0, h=10, txt=f'{file_type}: {img_name}', ln=1, align='L')
                            self.pdf.image(str(os.path.join(output_location, step, file)), x=0, y=20, w=WIDTH-30)
                            self.pdf.add_page()

            

        self.pdf.output(os.path.join(analysis_location, 'pdf_pipeline_summary.pdf'))


class SaveSpotDetectionResults(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, analysis_location, df_cellresults, df_spotresults, df_clusterresults, **kwargs):

        df_cellresults.to_csv(os.path.join(analysis_location, 'cell_results.csv'))
        df_spotresults.to_csv(os.path.join(analysis_location, 'spot_results.csv'))
        df_clusterresults.to_csv(os.path.join(analysis_location, 'cluster_results.csv'))

class SaveMasksToAnalysis(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, analysis_location, masks_nuclei:list[np.array] = None, masks_cytosol:list[np.array] = None, 
             masks_complete_cells:list[np.array] = None, map_id_imgprops:dict = None, **kwargs):

        if masks_nuclei is not None:
            with open(os.path.join(analysis_location, 'masks_nuclei.pkl'), 'wb') as f:
                pickle.dump(masks_nuclei, f)
        if masks_cytosol is not None:
            with open(os.path.join(analysis_location, 'masks_cytosol.pkl'), 'wb') as f:
                pickle.dump(masks_cytosol, f)
        if masks_complete_cells is not None:
            with open(os.path.join(analysis_location, 'masks_complete_cells.pkl'), 'wb') as f:
                pickle.dump(masks_complete_cells, f)
        if map_id_imgprops is not None:
            with open(os.path.join(analysis_location, 'map_id_imgprops.pkl'), 'wb') as f:
                pickle.dump(map_id_imgprops, f)


class SendAnalysisToNAS(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, analysis_location, initial_data_location, connection_config_location, share_name,   **kwargs):
        # where_to_send = str(os.path.dirname(initial_data_location))
        # print(where_to_send)
        # NASConnection(str(connection_config_location), share_name=str(share_name)).write_files_to_NAS(str(analysis_location), where_to_send)

        shutil.make_archive(analysis_location,'zip', pathlib.Path().absolute().joinpath(analysis_location))
        local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(analysis_location+'.zip')
        NASConnection(connection_config_location,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, initial_data_location)
        os.remove(pathlib.Path().absolute().joinpath(analysis_location+'.zip'))



class TrackPyAnlaysis(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, list_images, analysis_location, voxel_size_yx, voxel_size_z, timestep_s: float, df_spotresults: pd.DataFrame = None, 
             trackpy_features: pd.DataFrame = None, display_plots: bool = False, trackpy_link_distance_um: float = 1.25,
             link_search_range: list[float] = [1, 1.25, 1.5], trackpy_memory: int = 5, trackpy_max_lagtime: int = 25, **kwargs):

        # use bigfish or trackpy features
        if trackpy_features is None and df_spotresults is not None:
            trackpy_features = df_spotresults
            if 'x_subpx' in trackpy_features.columns:
                trackpy_features['xum'] = trackpy_features['x_subpx'] * voxel_size_yx / 1000 # convert px to um by px * nm/px * 1um/1000nm
                trackpy_features['yum'] = trackpy_features['y_subpx'] * voxel_size_yx / 1000
                trackpy_features['zum'] = trackpy_features['z_subpx'] * voxel_size_z / 1000
                trackpy_features['frame'] = trackpy_features['timepoint']
            else:
                trackpy_features['xum'] = trackpy_features['x_px'] * voxel_size_yx / 1000
                trackpy_features['yum'] = trackpy_features['y_px'] * voxel_size_yx / 1000
                trackpy_features['zum'] = trackpy_features['z_px'] * voxel_size_z / 1000
                trackpy_features['frame'] = trackpy_features['timepoint']

        elif trackpy_features is None and df_spotresults is None:
            raise ValueError('No spot detection results provided')

        else: 
            trackpy_features = trackpy_features

        # prealocate variables
        links = None
        diffusion_constants = {}
        msds = None

        # iterate over fovs
        for i, fov in enumerate(trackpy_features['fov'].unique()):
            features = trackpy_features[trackpy_features['fov'] == fov]

            if display_plots:
                for s in link_search_range:
                    if 'zum' in features.columns:
                        pos_columns = ['xum', 'yum', 'zum']
                    else:
                        pos_columns = ['xum', 'yum']
                    linked = tp.link_df(features, s, memory=trackpy_memory, pos_columns=pos_columns)
                    hist, bins = np.histogram(np.bincount(linked.particle.astype(int)),
                                            bins=np.arange(30))
                    plt.step(bins[1:], hist, label='range = {} microns'.format(s))
                plt.ylabel('relative frequency')
                plt.xlabel('track length (frames)')
                plt.legend()
                plt.show()

            # link features
            linked = tp.link_df(features, trackpy_link_distance_um, pos_columns=pos_columns)
            
            # calculate msd and diffusion constants
            fps = 1/timestep_s
            msd3D = tp.emsd(linked, mpp=(voxel_size_yx/1000, voxel_size_yx/1000, voxel_size_z/1000) if 'zum' in features.columns else (voxel_size_yx/1000, voxel_size_yx/1000),
                            fps=fps, max_lagtime=trackpy_max_lagtime,
                            pos_columns=pos_columns)
            slope = np.linalg.lstsq(msd3D.index[:, np.newaxis], msd3D)[0][0]

            if display_plots:
                ax = msd3D.plot(style='o', label='MSD in 3D')
                ax.plot(np.linspace(0, np.max(msd3D.index), 10), slope * np.linspace(0, np.max(msd3D.index), 10), label='linear fit')
                ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
                ax.legend(loc='upper left')
                plt.show()
                if 'zum' in features.columns:
                    print(r'The diffusion constant is {0:.2f} μm²/s'.format(slope / 6))
                else:
                    print(r'The diffusion constant is {0:.2f} μm²/s'.format(slope / 4))

            # merge dataframes from multiple fovs
            linked['fov'] = fov*len(linked)
            if links is None:
                links = linked
            else:
                links = pd.concat([links, linked])

            # merge dataframes from multiple fovs
            msd3D['fov'] = fov*len(msd3D)
            if msds is None:
                msds = msd3D    
            else:
                msds = pd.concat([msds, msd3D])

            # calculate diffusion constant for each fov in 2D or 3D
            if 'zum' in features.columns:
                diffusion_constants[f'fov num {fov}'] = slope / 6
            else:
                diffusion_constants[f'fov num {fov}'] = slope / 4

        if analysis_location is not None:
            links.to_csv(os.path.join(analysis_location,'trackpy_links.csv'))
            msd3D.to_csv(os.path.join(analysis_location,'trackpy_msd.csv'))
            # save diffusion constants to a text file
            with open(os.path.join(analysis_location,'diffusion_constants.txt'), 'w') as f:
                for key in diffusion_constants.keys():
                    f.write(f'{key}: {diffusion_constants[key]}\n')

        print('lagtime', trackpy_max_lagtime)

        return links, msds, diffusion_constants


        









        


        


class DeleteTempFiles(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, output_location, **kwargs):
        shutil.rmtree(output_location)













#%% Luis
class SavePDFReport(postPipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, initial_data_location, num_img_2_run, list_images,
             list_image_names, segmentation_successful, list_is_image_sharp, save_files, show_plots,
             temp_folder_name, dfFISH, FISHChannel, cytoChannel, nucChannel, masks_complete_cells, masks_nuclei,
             diameter_cytosol, diameter_nucleus, minimum_spots_cluster, CLUSTER_RADIUS, voxel_size_z,
             voxel_size_yx, spot_yx, spot_z, threshold_for_spot_detection, avg_number_of_spots_per_cell_each_ch,
             number_detected_cells,
             list_metric_sharpness_images,
             remove_out_of_focus_images, sharpness_threshold, remove_z_slices_borders, NUMBER_Z_SLICES_TO_TRIM, img_id,
             number_of_images_to_process, automatic_spot_detection_threshold, individual_threshold_spot_detection,
             save_pdf_report, list_z_slices_per_image,
             save_all_images, **kwargs):

        name_for_files = initial_data_location.name
        list_images = list_images[:num_img_2_run]
        list_files_names = list_image_names[:num_img_2_run]
        list_segmentation_successful = segmentation_successful[:num_img_2_run]
        list_is_image_sharp = list_is_image_sharp[:num_img_2_run]

        list_voxels = []
        list_spots = []
        for i in range(len(FISHChannel)):
            list_voxels.append([voxel_size_z, voxel_size_yx])
            list_spots.append([spot_z, spot_yx])

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
                     list_spots=list_spots,
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


# TODO: Complete the lower postPipelineSteps
class Luis_Additional_Plots(postPipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, initial_data_location,
             dfFISH, cytoChannel, nucChannel, FISHChannel, minimum_spots_cluster, output_identification_string,
             plot_for_pseudo_cytosol=False, save_pdf_report=True, **kwargs):

        list_files_distributions = Plots().plot_all_distributions(dfFISH, cytoChannel,
                                                                  nucChannel, FISHChannel,
                                                                  minimum_spots_cluster,
                                                                  output_identification_string)

        file_plots_bleed_thru = Plots().plot_scatter_bleed_thru(dfFISH, cytoChannel,
                                                                nucChannel, output_identification_string)

        # plots
        if not Utilities().is_None(cytoChannel):
            file_plots_int_ratio = Plots().plot_nuc_cyto_int_ratio_distributions(dfFISH,
                                                                                 output_identification_string=None,
                                                                                 plot_for_pseudo_cytosol=False)
        else:
            file_plots_int_ratio = None
        file_plots_int_pseudo_ratio = Plots().plot_nuc_cyto_int_ratio_distributions(dfFISH,
                                                                                    output_identification_string=None,
                                                                                    plot_for_pseudo_cytosol=True)

        ######################################
        ######################################
        # Saving data and plots, and sending data to NAS
        Utilities().save_output_to_folder(output_identification_string,
                                          initial_data_location,
                                          list_files_distributions=list_files_distributions,
                                          file_plots_bleed_thru=file_plots_bleed_thru,
                                          file_plots_int_ratio=file_plots_int_ratio,
                                          file_plots_int_pseudo_ratio=file_plots_int_pseudo_ratio,
                                          channels_with_FISH=FISHChannel,
                                          save_pdf_report=save_pdf_report)


class Send_Data_To_Nas(postPipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, output_identification_string,
             initial_data_location,
             path_to_config_file,
             path_to_masks_dir,
             diameter_nucleus,
             diameter_cytosol,
             send_data_to_NAS,
             masks_dir, **kwargs):
        # sending data to NAS
        analysis_folder_name, mask_dir_complete_name = Utilities().sending_data_to_NAS(output_identification_string,
                                                                                       initial_data_location,
                                                                                       path_to_config_file,
                                                                                       path_to_masks_dir,
                                                                                       diameter_nucleus,
                                                                                       diameter_cytosol,
                                                                                       send_data_to_NAS,
                                                                                       masks_dir)


class Move_Results_To_Analysis_Folder(postPipelineStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, output_identification_string,
             initial_data_location,
             save_filtered_images,
             diameter_nucleus,
             diameter_cytosol,
             local_or_NAS,
             save_masks_as_file:bool = False,
             save_files: bool = False,
             masks_dir=None, path_to_masks_dir=None, **kwargs):

        if path_to_masks_dir is None and save_masks_as_file and save_files:
            mask_folder_created_by_pipeline = 'masks_' + initial_data_location.name  # default name by pipeline
            name_final_masks = initial_data_location.name + '___nuc_' + str(diameter_nucleus) + '__cyto_' + str(
                diameter_cytosol)
            mask_dir_complete_name = 'masks_' + name_final_masks  # final name for masks dir
            shutil.move(mask_folder_created_by_pipeline, mask_dir_complete_name)  # remaing the masks dir
        elif masks_dir is None:
            mask_dir_complete_name = None
        else:
            mask_dir_complete_name = masks_dir.name

        Utilities().move_results_to_analyses_folder(output_identification_string, initial_data_location,
                                                    mask_dir_complete_name, path_to_masks_dir, save_filtered_images,
                                                    local_or_NAS, save_masks_as_file)






