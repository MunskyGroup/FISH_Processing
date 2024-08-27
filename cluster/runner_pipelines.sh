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

kwarg_location=$1

# ###################  PATHS TO CODE FILES  ############################
path_to_config_file="$HOME/FISH_Processing/config.yml"
path_to_executable="${PWD%/*}/src/pipeline_runner.py"

# ########### PYTHON PROGRAM #############################

output_names=""output__"${kwarg_location////__}"".txt"
 ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$kwarg_location" >> "$output_names" &
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
