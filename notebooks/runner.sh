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
folder_complete_path='Test/test_dir'
#folder_complete_path='smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_woDex'
#folder_complete_path='smFISH_images/Linda_smFISH_images/Confocal/20220117/GAPDH-Cy3_NFKBIA-Cy5_1h_100nMDex'
#folder_complete_path='smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_2h_100nMDex'
#folder_complete_path='smFISH_images/Linda_smFISH_images/Confocal/20220117/GAPDH-Cy3_NFKBIA-Cy5_4h_100nMDex'
send_data_to_NAS=0       # If data sent back to NAS use 1.
diamter_nucleus=120      # approximate nucleus size in pixels
diameter_cytosol=220     # approximate cytosol size in pixels
psf_z=330                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=110               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.

# Batch script instructions:
# https://www.baeldung.com/linux/use-command-line-arguments-in-bash-script 

# ########### PYTHON PROGRAM #############################
nohup python3 ./pipeline_executable.py  $folder_complete_path $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx >> out.txt &

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: bash -i shell_script.sh

# TODO LIST
# Create a runner function to test multiple parameters at the same time.
# Improve the plotting. 
# Create a new function to compare multiple and  plot multiple dataframes. 