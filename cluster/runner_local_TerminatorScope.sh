#!/bin/sh

# Bash script to run multiple python codes.
# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

# ########### ACTIVATE ENV #############################
# To load the env pass the specific location of the env and then activate it. 
# If not sure about the env location use: source activate <<venv_name>>   echo $CONDA_PREFIX
#source /home/"$USER"/anaconda3/envs/FISH_processing
conda activate FISH_processing
export CUDA_VISIBLE_DEVICES=0,1

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name  of the <<python_file.py>>, and  the rest are in positional order 
# Make sure to convert str to the desired data types.

# Paths with configuration files
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
path_to_config_file="$HOME/Desktop/config.yml"

# List or path to process
#list_test=('smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_75min')
list_Sawyer=(\
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_0min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_10min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_20min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_30min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_40min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_50min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230425_DUSP1_A549_DEX_60min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230426_DUSP1_A549_DEX_75min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230426_DUSP1_A549_DEX_90min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230426_DUSP1_A549_DEX_120min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230426_DUSP1_A549_DEX_150min_031123' \
'smFISH_images/Sawyer_smFISH_images/A549/Terminator/20230426_DUSP1_A549_DEX_180min_031123' )

# Software parameters
NUMBER_OF_CORES=1
diameter_nucleus=100                 # Approximate nucleus size in pixels
diameter_cytosol=200                 # Approximate cytosol size in pixels
psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=120                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=500                     # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=120                    # Microscope conversion px to nanometers in the xy axis.
channels_with_nucleus='None'         # Channel to pass to python for nucleus segmentation
channels_with_cytosol='[1]'          # Channel to pass to python for cytosol segmentation
channels_with_FISH='[0]'             # Channel to pass to python for spot detection
send_data_to_NAS=1                   # If data sent back to NAS use 1.
download_data_from_NAS=1             # Download data from NAS
path_to_masks_dir='None'             # 'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
save_all_images=0                    # If true, it shows a all planes for the FISH plot detection.
threshold_for_spot_detection='None'  # Thresholds for spot detection. Use an integer for a defined value, or 'None' for automatic detection.
save_filtered_images=0               #         
optimization_segmentation_method='default' # optimization_segmentation_method = 'default' 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
remove_z_slices_borders=1            # Use this flag to remove 2 z-slices from the top and bottom of the stack. This is needed to remove z-slices that are out of focus.
remove_out_of_focus_images=1         # Flag to remove out of focus images
save_pdf_report=0
# ######### Parameters to reformat images to standard format ########
convert_to_standard_format=1
number_color_channels=2
number_of_fov=1
# Use the following parameters if you have a metadatafile that contains the number of color channels and the number of FOVs
use_metadata=0
is_format_FOV_Z_Y_X_C=0
# ###########################################################

# ########### PYTHON PROGRAM #############################
for folder in ${list_Sawyer[*]}; do
     output_names=""output__"${folder////__}"".txt"
     nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images $remove_z_slices_borders $remove_out_of_focus_images $save_pdf_report $convert_to_standard_format $number_color_channels $number_of_fov $use_metadata $is_format_FOV_Z_Y_X_C >> "$output_names" &
     wait
done
# #####################################
# #####################################

# Deactivating the environment
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner_local_TerminatorScope.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3
# ps -ef | grep python3 | grep "pipeline_"
# ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')


# nvidia-smi | grep 'Default'

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif

exit 0