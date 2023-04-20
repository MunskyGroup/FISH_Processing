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
list_test=('smFISH_images/Eric_smFISH_images/20230306/DUSP1_0nM_Dex_0min_012623')

# Software parameters
NUMBER_OF_CORES=1
diameter_nucleus=125                 # Approximate nucleus size in pixels
diameter_cytosol=225                 # Approximate cytosol size in pixels
psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=160                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=500                     # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=160                    # Microscope conversion px to nanometers in the xy axis.
#channels_with_nucleus='[0]'          # Channel to pass to python for nucleus segmentation
#channels_with_cytosol='[2]'          # Channel to pass to python for cytosol segmentation
#channels_with_FISH='[1]'             # Channel to pass to python for spot detection
channels_with_nucleus='[2]'                  # Channel to pass to python for nucleus segmentation
channels_with_cytosol='[1]'                 # Channel to pass to python for cytosol segmentation
channels_with_FISH='[0]'                   # Channel to pass to python for spot detection
send_data_to_NAS=0                   # If data sent back to NAS use 1.
download_data_from_NAS=1             # Download data from NAS
path_to_masks_dir='None'             # 'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
save_all_images=0                    # If true, it shows a all planes for the FISH plot detection.
threshold_for_spot_detection='None'  # Thresholds for spot detection. Use an integer for a defined value, or 'None' for automatic detection.
save_filtered_images=0               #         
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None

# ########### PYTHON PROGRAM #############################
for folder in ${list_test[*]}; do
     output_names=""output__"${folder////__}"".txt"
     nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
     wait
done
# #####################################
# #####################################

# Deactivating the environment
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner_local.sh /dev/null 2>&1 & disown

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