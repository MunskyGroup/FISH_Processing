#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:00:00 2022

@author: luis_aguilera
"""

# Importing libraries
import sys
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import pathlib
import warnings
import shutil
import zipfile
import os
import json
warnings.filterwarnings("ignore")
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(np.random.randint(0,2,1))
######################################
## User passed arguments
remote_folder = sys.argv[1]                             # Path to the remote Folder
data_folder_path = pathlib.Path(remote_folder)
send_data_to_NAS = int(sys.argv[2])                     # Flag to send data back to NAS
diameter_nucleus = int(sys.argv[3])                      # approximate nucleus size in pixels
diameter_cytosol = int(sys.argv[4])  #250               # approximate cytosol size in pixels
psf_z_1 = int(sys.argv[5])       #350                   # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx_1 = int(sys.argv[6])      #150                   # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel= int(sys.argv[7])                       # Channel to pass to python for nucleus segmentation
cyto_channel= json.loads(sys.argv[8])                        # Channel to pass to python for cytosol segmentation
FISH_channel= int(sys.argv[9])                          # Channel to pass to python for spot detection
FISH_second_channel= sys.argv[10]                  # Channel to pass to python for spot detection
if FISH_second_channel == 'None':
  FISH_second_channel = None
else:
  FISH_second_channel= int(FISH_second_channel)
output_name = sys.argv[11]                              # Output file name
# Path to credentials
path_to_config_file = pathlib.Path(sys.argv[12])
download_data_from_NAS= int(sys.argv[13])
path_to_masks_dir= sys.argv[14]
if path_to_masks_dir == 'None':
  path_to_masks_dir = None
else:
  path_to_masks_dir = pathlib.Path(path_to_masks_dir )
optimization_segmentation_method= pathlib.Path(sys.argv[15])
if optimization_segmentation_method == 'None':
  optimization_segmentation_method = None

save_all_images=int(sys.argv[16])

# Deffining directories
current_dir = pathlib.Path().absolute()
fa_dir = current_dir.parents[0].joinpath('src')

# Importing fish_analyses module
sys.path.append(str(fa_dir))
import fish_analyses as fa

# Printing banner
fa.Banner().print_banner()
share_name = 'share'

# names for final folders
name_final_folder = data_folder_path.name +'___nuc_' + str(diameter_nucleus) +'__cyto_' + str(diameter_cytosol) +'__psfz_' + str(psf_z_1) +'__psfyx_' + str(psf_yx_1)
name_final_masks = data_folder_path.name +'___nuc_' + str(diameter_nucleus) + '__cyto_' + str(diameter_cytosol) 

def download_data_NAS(path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200):
  # Downloading data from NAS
  local_folder_path = pathlib.Path().absolute().joinpath('temp_' + data_folder_path.name)
  fa.NASConnection(path_to_config_file,share_name = share_name).copy_files(data_folder_path, local_folder_path,timeout=timeout)
  local_data_dir = local_folder_path     # path to a folder with images.
  # Downloading masks from NAS
  if not (path_to_masks_dir is None):
    local_folder_path_masks = pathlib.Path().absolute().joinpath( path_to_masks_dir.stem  )
    zip_file_path = local_folder_path_masks.joinpath( path_to_masks_dir.stem +'.zip')
    print(zip_file_path)
    fa.NASConnection(path_to_config_file,share_name = share_name).download_file(path_to_masks_dir, local_folder_path_masks,timeout=timeout)
    # Unzip downloaded images and update mask directory
    file_to_unzip = zipfile.ZipFile(str(zip_file_path)) # opens zip
    # Iterates for each file in zip file
    for file_in_zip in file_to_unzip.namelist():
      # Extracts data to specific folder
      file_to_unzip.extract(file_in_zip,local_folder_path_masks)
    # Closes the zip file
    file_to_unzip.close()
    # removes the original zip file
    os.remove(zip_file_path)
    masks_dir = local_folder_path_masks
  else:
    masks_dir = None
  return local_data_dir, masks_dir

# Download data from NAS
if download_data_from_NAS == True:
  share_name = 'share'
  local_data_dir, masks_dir= download_data_NAS(path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200)
else:
  local_data_dir = data_folder_path 
  masks_dir = path_to_masks_dir 

# Parameters for the code
if isinstance(cyto_channel, list):
  channels_with_cytosol = [cyto_channel]            # list or int indicating the channels where the cytosol is detectable
else:
  channels_with_cytosol = [cyto_channel]            # list or int indicating the channels where the cytosol is detectable

channels_with_nucleus = nucleus_channel                # list or int indicating the channels where the nucleus is detectable

# Deffining FISH Channels
if (FISH_second_channel==0) or (FISH_second_channel==None):
    channels_with_FISH = [FISH_channel]               # list or int with the channels with FISH spots that are used for the quantification
else:
    channels_with_FISH = [FISH_channel,FISH_second_channel ]

# Detecting if images need to be merged
is_needed_to_merge_images = fa.MergeChannels(local_data_dir, substring_to_detect_in_file_name = '.*_C0.tif', save_figure =1).checking_images()
if is_needed_to_merge_images == True:
  list_file_names, list_images, number_images, output_to_path = fa.MergeChannels(local_data_dir, substring_to_detect_in_file_name = '.*_C0.tif', save_figure =1).merge()
  local_data_dir_un_merged =local_data_dir
  local_data_dir = local_data_dir.joinpath('merged')

# Parameters for FISH detection
voxel_size_z = 500                       # Microscope conversion px to nanometers in the z axis.
voxel_size_yx = 103                      # Microscope conversion px to nanometers in the xy axis.

list_voxels = [ [voxel_size_z,voxel_size_yx  ]  ]
list_psfs = [ [psf_z_1, psf_yx_1] ]
# Cluster Detection
minimum_spots_cluster = 2                # The number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
show_plots=True                          # Flag to display plots

# Running the pipeline
dataframe_FISH,_,_,_ = fa.PipelineFISH(local_data_dir, channels_with_cytosol, channels_with_nucleus, channels_with_FISH,diameter_nucleus, diameter_cytosol, minimum_spots_cluster, masks_dir=masks_dir,  list_voxels=list_voxels, list_psfs=list_psfs, show_plot=show_plots, file_name_str =data_folder_path.name, optimization_segmentation_method = optimization_segmentation_method,save_all_images=save_all_images ).run()

# Number of cells
spot_type_selected = 0
number_cells = dataframe_FISH['cell_id'].nunique()
print(number_cells)
# Number of spots
number_of_spots_per_cell = [len( dataframe_FISH.loc[  (dataframe_FISH['cell_id']==i)  & (dataframe_FISH['spot_type']==spot_type_selected) ].spot_id) for i in range(0, number_cells)]
# Number of spots in cytosol
number_of_spots_per_cell_cytosol = [len( dataframe_FISH.loc[  (dataframe_FISH['cell_id']==i) & (dataframe_FISH['is_nuc']==False) & (dataframe_FISH['spot_type']==spot_type_selected) ].spot_id) for i in range(0, number_cells)]
# Number of spots in nucleus
number_of_spots_per_cell_nucleus = [len( dataframe_FISH.loc[  (dataframe_FISH['cell_id']==i) &  (dataframe_FISH['is_cluster']==False) & (dataframe_FISH['is_nuc']==True) & (dataframe_FISH['spot_type']==spot_type_selected)    ].spot_id) for i in range(0, number_cells)]
# Number of TS per cell.
number_of_TS_per_cell = [len( dataframe_FISH.loc[  (dataframe_FISH['cell_id']==i) &  (dataframe_FISH['is_cluster']==True) & (dataframe_FISH['is_nuc']==True) & (dataframe_FISH['spot_type']==spot_type_selected) & (dataframe_FISH['cluster_size'] >=4) ].spot_id) for i in range(0, number_cells)]
# Number of RNA in a TS
ts_size =  dataframe_FISH.loc[   (dataframe_FISH['is_cluster']==True) & (dataframe_FISH['is_nuc']==True)  & (dataframe_FISH['spot_type']==spot_type_selected)   ].cluster_size.values
# Size of each cell
cell_size = dataframe_FISH.loc[  (dataframe_FISH['spot_id']==0)  ].cell_area_px.values

# Plotting intensity distributions
plt.style.use('ggplot')  # ggplot  #default
def plot_probability_distribution(data_to_plot, numBins = 10, title='', xlab='', ylab='', color='r', subplots=False, show_grid=True, fig=plt.figure() ):
  n, bins, _ = plt.hist(data_to_plot,bins=numBins,density=False,color=color)
  plt.xlabel(xlab, size=16)
  plt.ylabel(ylab, size=16)
  plt.grid(show_grid)
  plt.text(bins[(len(bins)//2)],(np.amax(n)//2).astype(int),'mean = '+str(round( np.mean(data_to_plot) ,1) ), fontsize=14,bbox=dict(facecolor='w', alpha=0.5) )
  plt.title(title, size=16)
  return (f)

#Plotting
fig_size = (30, 7)
f = plt.figure(figsize=fig_size)
#ylab='Probability'
ylab='Frequency Count'  
# adding subplots
f.add_subplot(1,5,1) 
plot_probability_distribution( number_of_spots_per_cell, numBins=20,  title='Total Num Spots per cell', xlab='Number', ylab=ylab, fig=f, color='orangered')
f.add_subplot(1,5,2) 
plot_probability_distribution(number_of_spots_per_cell_cytosol,   numBins=20,  title='Num Spots in Cytosol', xlab='Number', ylab=ylab, fig=f, color='orangered')
f.add_subplot(1,5,3) 
plot_probability_distribution(number_of_spots_per_cell_nucleus, numBins=20,    title='Num Spots in Nucleus', xlab='Number', ylab=ylab, fig=f, color='orangered')
f.add_subplot(1,5,4) 
plot_probability_distribution(ts_size, numBins=20,    title='Clusters in nucleus', xlab='RNA per Cluster', ylab=ylab, fig=f, color='orangered')
f.add_subplot(1,5,5) 
plot_probability_distribution(number_of_TS_per_cell ,  numBins=20, title='Number TS per cell', xlab='[TS (>= 4 rna)]', ylab=ylab, fig=f, color='orangered')
plt.savefig('plots_'+data_folder_path.name+'.png')
plt.show()

# create results folder
if not os.path.exists(str('analysis_'+ name_final_folder)):
  os.makedirs(str('analysis_'+ name_final_folder))    

#figure_path 
pathlib.Path().absolute().joinpath('plots_'+ data_folder_path.name +'.png').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ name_final_folder),'plots_'+ data_folder_path.name +'.png'))
#metadata_path
pathlib.Path().absolute().joinpath('metadata_'+ data_folder_path.name +'.txt').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ name_final_folder),'metadata_'+ data_folder_path.name +'.txt'))
#dataframe_path 
pathlib.Path().absolute().joinpath('dataframe_' + data_folder_path.name +'.csv').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ name_final_folder),'dataframe_'+ data_folder_path.name +'.csv'))
#pdf_path 
pathlib.Path().absolute().joinpath('pdf_report_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ name_final_folder),'pdf_report_'+ data_folder_path.name +'.pdf'))
# copy output file
shutil.copyfile(pathlib.Path().absolute().joinpath(output_name),    pathlib.Path().absolute().joinpath(str('analysis_'+ name_final_folder), output_name) )

# Writing analyses data to NAS
analysis_folder_name = 'analysis_'+ name_final_folder
if send_data_to_NAS == True:
  shutil.make_archive(analysis_folder_name,'zip',pathlib.Path().absolute().joinpath(analysis_folder_name))
  local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip')
  fa.NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
  os.remove(pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip'))
  # Delete temporal images downloaded from NAS
  shutil.rmtree(local_data_dir)

# Writing masks to NAS
if path_to_masks_dir == None: 
  mask_folder_created_by_pipeline = 'masks_'+ data_folder_path.name # default name by pipeline
  name_final_masks = data_folder_path.name +'___nuc_' + str(diameter_nucleus) + '__cyto_' + str(diameter_cytosol) 
  mask_dir_complete_name = 'masks_'+ name_final_masks # final name for masks dir
  shutil.move(mask_folder_created_by_pipeline, mask_dir_complete_name ) # remaing the masks dir
else: 
  mask_dir_complete_name = masks_dir.name
    
if (send_data_to_NAS == True) and (path_to_masks_dir == None) :
  shutil.make_archive( mask_dir_complete_name , 'zip', pathlib.Path().absolute().joinpath(mask_dir_complete_name))
  local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip')
  fa.NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
  os.remove(pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip'))

# Moving all results to "analyses" folder
if not os.path.exists(str('analyses')):
  os.makedirs(str('analyses'))

# Subfolder name
final_dir_name =pathlib.Path().absolute().joinpath('analyses', analysis_folder_name)

# Removing directory if exist
if os.path.exists(str(final_dir_name)):
  shutil.rmtree(str(final_dir_name))

# Movng results to a subdirectory in 'analyses' folder
pathlib.Path().absolute().joinpath(analysis_folder_name).rename(final_dir_name )

# Moving masks to a subdirectory in 'analyses' folder
if (download_data_from_NAS == True) or (path_to_masks_dir == None):
  final_mask_dir_name = pathlib.Path().absolute().joinpath('analyses', mask_dir_complete_name)
  if os.path.exists(str(final_mask_dir_name)):
    shutil.rmtree(str(final_mask_dir_name))
  pathlib.Path().absolute().joinpath(mask_dir_complete_name).rename(final_mask_dir_name )

# Delete local temporal files
temp_results_folder_name = pathlib.Path().absolute().joinpath('temp_results_' + data_folder_path.name)
shutil.rmtree(temp_results_folder_name)

if (download_data_from_NAS == True):
    # Delete temporal images downloaded from NAS
    shutil.rmtree(local_data_dir)
    if is_needed_to_merge_images == True:
      shutil.rmtree(local_data_dir_un_merged)
      
# remove output_file
os.remove(pathlib.Path().absolute().joinpath(output_name))