#!/bin/bash
#SBATCH --gres=gpu:4
# #SBATCH --nodelist=gpu2    # gpu2 gpu3 gpu4
#SBATCH --partition=all
#SBATCH --ntasks=4
#SBATCH --job-name=t2

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

echo "Starting my job..."
# Start timing the process
start_time=$(date +%s)

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>

#list_test=('smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_75min')
list_test=($1)

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

NUMBER_OF_CORES=4

# ###################  PATHS TO CODE FILES  ############################
path_to_config_file="$HOME/FISH_Processing/config.yml"
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 

# ###################  CODE PARAMETERS ############################
diameter_nucleus=100                 # Approximate nucleus size in pixels
diameter_cytosol=200                 # Approximate cytosol size in pixels
psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=130                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=$5                     # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=130                    # Microscope conversion px to nanometers in the xy axis.
channels_with_nucleus=$2   #'[2]'                 # Channel to pass to python for nucleus segmentation
channels_with_cytosol=$3   #'[1]'                 # Channel to pass to python for cytosol segmentation
channels_with_FISH=$4      #'[0]'                 # Channel to pass to python for spot detection
send_data_to_NAS=1                   # If data sent back to NAS use 1.
download_data_from_NAS=1             # Download data from NAS
path_to_masks_dir='None'             # 'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
save_all_images=0                    # If true, it shows a all planes for the FISH plot detection.
threshold_for_spot_detection='None'  # Thresholds for spot detection. Use an integer for a defined value, or 'None' for automatic detection.
save_filtered_images=0               #         
optimization_segmentation_method='None' # optimization_segmentation_method = 'default' 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
remove_z_slices_borders=1       # Use this flag to remove 2 z-slices from the top and bottom of the stack. This is needed to remove z-slices that are out of focus.
remove_out_of_focus_images=1         # Flag to remove out of focus images
save_pdf_report=1
# ######### Parameters to reformat images to standard format ########
convert_to_standard_format=1
number_color_channels=3
number_of_fov=70
# Use the following parameters to automatically detect the image shape
use_metadata=1
is_format_FOV_Z_Y_X_C=1
# #####################################################################

# ########### PYTHON PROGRAM #############################
for folder in ${list_test[*]}; do
     output_names=""output__"${folder////__}"".txt"
     ~/miniconda3/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images $remove_z_slices_borders $remove_out_of_focus_images $save_pdf_report $convert_to_standard_format $number_color_channels $number_of_fov $use_metadata $is_format_FOV_Z_Y_X_C >> "$output_names" &
     wait
done

# ########### PYTHON PROGRAM USING DIR FOR MASKS #############################
#counter=0
#for folder in ${list_test[*]}; do
#     output_names=""output__"${folder////__}"".txt"
#     path_to_masks_dir="${mask_list[counter]}"
#     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images >> "$output_names" &
#     ((counter++))
#     wait
#done
# End timing the process

end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster.sh location '[0]' '[1]' '[2]' 500 /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* masks_* 

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue
