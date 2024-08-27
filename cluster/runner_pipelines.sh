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

# ###################  PATHS TO CODE FILES  ############################
path_to_config_file="$HOME/FISH_Processing/config.yml"
path_to_executable="${PWD%/*}/src/pipeline_runner.py"

# ########### PYTHON PROGRAM #############################

 ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images $remove_z_slices_borders $remove_out_of_focus_images $save_pdf_report $convert_to_standard_format $number_color_channels $number_of_fov $use_metadata $is_format_FOV_Z_Y_X_C >> "$output_names" &
 wait


# End timing the process
end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster.sh /dev/null 2>&1 & disown

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
