#!/bin/sh

# Bash script to run multiple python codes.
# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

# ########### ACTIVATE ENV #############################
# To load the env pass the specific location of the env and then activate it. 
# If not sure about the env location use: source activate <<venv_name>>   echo $CONDA_PREFIX
#source /home/luisub/anaconda3/envs/FISH_processing
#conda activate FISH_processing

source /top/college/academic/ChemE/"$USER"/home/.conda/envs/FISH_processing
conda activate FISH_processing

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

# Declare a string array
folder="test"
send_data_to_NAS="0"       # If data sent back to NAS use 1.
diamter_nucleus="120"      # approximate nucleus size in pixels
diameter_cytosol="220"     # approximate cytosol size in pixels
psf_z="350"                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx="120"               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
send_data_to_NAS="0"
nohup /top/college/academic/ChemE/"$USER"/home/.conda/envs/FISH_processing/bin/python ./pipeline_local.py $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx >> output.txt


# ########### PYTHON PROGRAM #############################
#for folder in ${folders[*]}; do
#     #nohup python3 ./pipeline_local.py  $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx >> output.txt
#     nohup /top/college/academic/ChemE/"$USER"/home/.conda/envs/FISH_processing/bin/python ./pipeline_local.py  $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx >> output.txt
#     wait
#done
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: bash runner_cluster.sh &

exit 0