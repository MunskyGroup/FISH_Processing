#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:00:00 2022

@author: luis_aguilera
"""


######################################
######################################
# Importing libraries
import sys
import pathlib
import warnings
import json
warnings.filterwarnings("ignore")
tuple_none = (None, 'None', 'none',['None'],['none'],[None])
######################################
######################################


######################################
######################################
# Defining directories
current_dir = pathlib.Path().absolute()
fa_dir = current_dir.parents[0].joinpath('src')
# Importing fish_analyses module
sys.path.append(str(fa_dir))
import fish_analyses as fa
# Printing banner
fa.Banner().print_banner()
share_name = 'share'
######################################
######################################


######################################
######################################
## User passed arguments
remote_folder = sys.argv[1]                              # Path to the remote Folder
data_folder_path = pathlib.Path(remote_folder)           # Path to folder
send_data_to_NAS = int(sys.argv[2])                      # Flag to send data back to NAS
diameter_nucleus = int(sys.argv[3])                      # Approximate nucleus size in pixels
diameter_cytosol = int(sys.argv[4])                      # Approximate cytosol size in pixels
voxel_size_z  = int(sys.argv[5])                         # Microscope conversion px to nanometers in the z axis.
voxel_size_yx  = int(sys.argv[6])                        # Microscope conversion px to nanometers in the xy axis.
psf_z = int(sys.argv[7])                                 # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx = int(sys.argv[8])                                # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
# Segmentation Channels
if sys.argv[9] in tuple_none:
    channels_with_nucleus = None
else:
    channels_with_nucleus= json.loads(sys.argv[9])       # Channel to pass to python for nucleus segmentation
if sys.argv[10] in tuple_none:
    channels_with_cytosol = None
else:
    channels_with_cytosol = json.loads(sys.argv[10])     # Channel to pass to python for cytosol segmentation
if channels_with_cytosol in tuple_none:
    channels_with_cytosol = None
if channels_with_nucleus in tuple_none:
    channels_with_nucleus = None
# Parameters for the code
if not isinstance(channels_with_cytosol, list) and not (channels_with_cytosol is None):
    channels_with_cytosol = [channels_with_cytosol]            # list or int indicating the channels where the cytosol is detectable
if not isinstance(channels_with_nucleus, list) and not (channels_with_nucleus is None):
    channels_with_nucleus = [channels_with_nucleus]            # list or int indicating the channels where the nucleus is detectable
# FISH Channels
channels_with_FISH =json.loads(sys.argv[11])
if not isinstance(channels_with_FISH, list):
    channels_with_FISH = [channels_with_FISH]            # list or int indicating the channels where the cytosol is detectable
# Output file name
output_name = sys.argv[12]  
# Path to credentials
path_to_config_file = pathlib.Path(sys.argv[13])
download_data_from_NAS= int(sys.argv[14])
path_to_masks_dir= sys.argv[15]
# Path to directory with masks
if path_to_masks_dir in tuple_none:
    path_to_masks_dir = None
else:
    path_to_masks_dir = pathlib.Path(path_to_masks_dir )
optimization_segmentation_method= sys.argv[16]
# Additional parameters
if optimization_segmentation_method in tuple_none:
    optimization_segmentation_method = None
save_all_images=int(sys.argv[17])
# converting the threshold_for_spot_detection to a list to iterate for each FISH channel
if sys.argv[18] in tuple_none:
    threshold_for_spot_detection = None
else:
    threshold_for_spot_detection = json.loads(sys.argv[18])  
#list_threshold_for_spot_detection = fa.Utilities.create_list_thresholds_FISH(channels_with_FISH,threshold_for_spot_detection)
NUMBER_OF_CORES=int(sys.argv[19])
save_filtered_images = int(sys.argv[20])
remove_z_slices_borders = int(sys.argv[21])

####################################################################
########## Parameters to reformat images to standard format ########
convert_to_standard_format = int(sys.argv[22])
number_color_channels = int(sys.argv[23])
number_of_fov = int(sys.argv[24])
######################################
######################################
####################################################################


######################################
######################################
minimum_spots_cluster = 4                # The number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
number_of_images_to_process = None       # This section allows the user to select a subset of images to process. Use an integer to indicate the n images to process.
show_plots=False
######################################
######################################


######################################
######################################
# Download data from NAS
if convert_to_standard_format == False:
    local_data_dir, masks_dir, _, _, _, _ = fa.Utilities.read_images_from_folder( path_to_config_file, 
                                                                                    data_folder_path, 
                                                                                    path_to_masks_dir,  
                                                                                    download_data_from_NAS)
else:
    local_data_dir,masks_dir, _, _, _= fa.Utilities.convert_to_standard_format(data_folder_path=data_folder_path, 
                                                                                    path_to_config_file=path_to_config_file, 
                                                                                    download_data_from_NAS = download_data_from_NAS,
                                                                                    number_color_channels=number_color_channels,
                                                                                    number_of_fov=number_of_fov )
# Running the pipeline
dataframe_FISH,_,_,_,output_identification_string = fa.PipelineFISH(local_data_dir, 
                                                                    channels_with_cytosol, 
                                                                    channels_with_nucleus, 
                                                                    channels_with_FISH,
                                                                    diameter_nucleus, 
                                                                    diameter_cytosol, 
                                                                    minimum_spots_cluster, 
                                                                    masks_dir = masks_dir,  
                                                                    voxel_size_z = voxel_size_z,
                                                                    voxel_size_yx = voxel_size_yx, 
                                                                    psf_z = psf_z,
                                                                    psf_yx = psf_yx, 
                                                                    show_plots = show_plots,  
                                                                    file_name_str = data_folder_path.name, 
                                                                    optimization_segmentation_method = optimization_segmentation_method,
                                                                    save_all_images = save_all_images,
                                                                    threshold_for_spot_detection = threshold_for_spot_detection,
                                                                    NUMBER_OF_CORES = NUMBER_OF_CORES,
                                                                    save_filtered_images = save_filtered_images,
                                                                    number_of_images_to_process = number_of_images_to_process,
                                                                    remove_z_slices_borders=remove_z_slices_borders).run()
######################################
######################################


######################################
######################################

list_files_distributions = fa.Plots.plot_all_distributions (dataframe_FISH,channels_with_cytosol, channels_with_nucleus,channels_with_FISH,minimum_spots_cluster,output_identification_string )
file_plots_bleed_thru = fa.Plots.plot_scatter_bleed_thru(dataframe_FISH, channels_with_cytosol, channels_with_nucleus,output_identification_string)

######################################
######################################
# Saving data and plots, and sending data to NAS
fa.Utilities.save_output_to_folder(output_identification_string, data_folder_path, list_files_distributions=list_files_distributions, file_plots_bleed_thru=file_plots_bleed_thru,channels_with_FISH=channels_with_FISH)
# sending data to NAS
analysis_folder_name, mask_dir_complete_name = fa.Utilities.sending_data_to_NAS(output_identification_string, data_folder_path, path_to_config_file, path_to_masks_dir, diameter_nucleus, diameter_cytosol, send_data_to_NAS, masks_dir)
# Moving the complete analysis folder to final analyses folder 
fa.Utilities.move_results_to_analyses_folder( output_identification_string, data_folder_path, mask_dir_complete_name, path_to_masks_dir, save_filtered_images, download_data_from_NAS )
######################################
######################################



