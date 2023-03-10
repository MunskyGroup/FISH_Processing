#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=testing_2

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>

list_test=('smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_75min')

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 


NUMBER_OF_CORES=4

path_to_config_file="$HOME/FISH_Processing/config.yml"
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 

diameter_nucleus=90                 # Approximate nucleus size in pixels
diameter_cytosol=180                 # Approximate cytosol size in pixels
psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=160                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=500                     # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=160                    # Microscope conversion px to nanometers in the xy axis.
nucleus_channel='[0]'                # Channel to pass to python for nucleus segmentation
cyto_channel='[2]'                   # Channel to pass to python for cytosol segmentation
FISH_channel='[1]'                   # Channel to pass to python for spot detection
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
     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
     wait
done

# ########### PYTHON PROGRAM USING DIR FOR MASKS #############################
#counter=0
#for folder in ${list_test[*]}; do
#     output_names=""output__"${folder////__}"".txt"
#     path_to_masks_dir="${mask_list[counter]}"
#     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images >> "$output_names" &
#     ((counter++))
#     wait
#done

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* 

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue
