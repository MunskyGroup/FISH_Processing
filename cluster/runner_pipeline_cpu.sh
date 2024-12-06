#!/bin/bash
#SBATCH --gres=gpu:4
# #SBATCH --nodelist=gpu3    # gpu2 gpu3 gpu4
# #SBATCH --exclude=gpu1,gpu2
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --ntasks=4
#SBATCH --job-name=t2

# module purge
module load gnu9/9.4.0
module load cudnn/8.3-10.2

echo "Starting my job..."

# Start timing the process
start_time=$(date +%s)

kwarg_location=$1
path_to_executable="${PWD%/*}/src/pipeline_executable.py"

# ########### PYTHON PROGRAM #############################
output_names=""output__"${kwarg_location}"".txt"
source ../.venv/bin/activate
python "$path_to_executable" "$kwarg_location" >> "$output_names" &
wait


# End timing the process
end_time=$(date +%s)
total_time=$(( (end_time - start_time) / 60 ))

# Print the time to complete the process
echo "Total time to complete the job: $total_time minutes"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_pipeline.sh /dev/null 2>&1 & disown

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
