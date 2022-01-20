#!/bin/sh

# Bash script to run multiple python codes.
# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>
# use nohup to run on background.

# ########### ACTIVATE ENV #############################
# To load the env pass the specific location of the env and then activate it. 
# If not sure about the env location use: source activate <<venv_name>>   echo $CONDA_PREFIX
source /home/luisub/anaconda3/envs/FISH_processing
conda activate FISH_processing

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 
# Make sure to convert str to the desired data types.

folder_complete_path='smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_2h_100nMDex'
merge_images=0

# Batch script instructions:
# https://www.baeldung.com/linux/use-command-line-arguments-in-bash-script 

# ########### PYTHON PROGRAM #############################
#nohup python3 ./pipeline_executable.py >> out.txt
#nohup python3 ./test.py  $folder_complete_path $merge_images $merge_images  >> out.txt &
nohup python3 ./pipeline_executable.py  $folder_complete_path $merge_images $merge_images  >> out.txt &

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: bash -i shell_script.sh


# Create a complex string for the final file name including the parameters used. Use this for the final zip file.