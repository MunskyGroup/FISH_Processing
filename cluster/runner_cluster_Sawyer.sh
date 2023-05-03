#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu4    # gpu2 gpu3 gpu4
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
diameter_cytosol=230                 # Approximate cytosol size in pixels
psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=105                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=500                     # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=96                    # Microscope conversion px to nanometers in the xy axis.
channels_with_nucleus='None'                  # Channel to pass to python for nucleus segmentation
channels_with_cytosol='[1]'                 # Channel to pass to python for cytosol segmentation
channels_with_FISH='[0]'                   # Channel to pass to python for spot detection
send_data_to_NAS=0                   # If data sent back to NAS use 1.
download_data_from_NAS=1             # Download data from NAS
path_to_masks_dir='None'             # 'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
save_all_images=0                    # If true, it shows a all planes for the FISH plot detection.
threshold_for_spot_detection='None'  # Thresholds for spot detection. Use an integer for a defined value, or 'None' for automatic detection.
save_filtered_images=0               #         
optimization_segmentation_method='default' # optimization_segmentation_method = 'default' 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
remove_z_slices_borders=False        # Use this flag to remove 2 z-slices from the top and bottom of the stack. This is needed to remove z-slices that are out of focus.

# ######### Parameters to reformat images to standard format ########
convert_to_standard_format=1
number_color_channels=0
number_of_fov=0
# #####################################################################


# ########### PYTHON PROGRAM #############################
for folder in ${list_Sawyer[*]}; do
     output_names=""output__"${folder////__}"".txt"
     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images $remove_z_slices_borders $convert_to_standard_format $number_color_channels $number_of_fov >> "$output_names" &
     wait
done
# ###########################################################
end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster_Sawyer.sh /dev/null 2>&1 & disown

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
