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
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 
# Make sure to convert str to the desired data types.
#folder_complete_path='Test/test_dir'
#folder_complete_path_0='Test/test_dir'
#folder_complete_path_1='Test/test_dir'

# To enable GPUs on code
# https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on

# Declare a string array

list_test=(\
'Test/test_dir' \
'Test/test_dir1' \
) 

list_ILB_Rep1=( \
'smFISH_images/Linda_smFISH_images/Confocal/20220125/GAPDH-Cy3_NFKBIA-Cy5_WO_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220203/GAPDH-Cy3_NFKBIA-Cy5_5min_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220127/GAPDH-Cy3_NFKBIA-Cy5_10min_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220125/GAPDH-Cy3_NFKBIA-Cy5_15min_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220125/GAPDH-Cy3_NFKBIA-Cy5_20min_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_30min_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_1h_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_2h_10ng_mL_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_3h_10ng_mL_IL-1B' ) 

list_ILB_Rep2=( \
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_WO_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_5min_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220209/GAPDH-Cy3_NFKBIA-Cy5_10min_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220209/GAPDH-Cy3_NFKBIA-Cy5_15min_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220209/GAPDH-Cy3_NFKBIA-Cy5_20min_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220207/GAPDH-Cy3_NFKBIA-Cy5_30min_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220207/GAPDH-Cy3_NFKBIA-Cy5_1h_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220207/GAPDH-Cy3_NFKBIA-Cy5_2h_10ng_mL_IL-1B_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220207/GAPDH-Cy3_NFKBIA-Cy5_3h_10ng_mL_IL-1B_Rep2' ) 


send_data_to_NAS=1       # If data sent back to NAS use 1.
diamter_nucleus=120      # approximate nucleus size in pixels
diameter_cytosol=220     # approximate cytosol size in pixels
psf_z=350                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=120               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel=0        # Channel to pass to python for nucleus segmentation
cyto_channel=2           # Channel to pass to python for cytosol segmentation
FISH_channel=1           # Channel to pass to python for spot detection
FISH_second_channel=0    # Channel to pass to python for spot detection in a second Channel, if 0 is ignored.
path_to_config_file="$HOME/Desktop/config.yml"

#########for loop
# over different parameters above
# pick ones with most cells, 3-4 for for these folders
#look for which has what effect on the output files (spot detection etc.)


#for folder in ${list_DUSP1_DEX[*]}; do
     #folder_new="${folder//\\//}"
     #output_name=''output__"${folder////__}"".txt"
     #nohup python3 ./pipeline_executable.py  $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx $nucleus_channel $cyto_channel $FISH_channel $FISH_second_channel $output_name "$path_to_config_file" >> $output_name &
#done

# var_new="${var//\\//}"
# // replace every
# \\ backslash
# / with
# / slash

# ########### PYTHON PROGRAM #############################
COUNTER=0
for folder in ${list_ILB_Rep2[*]}; do
     folder_new="${folder//\\//}"
     output_name=''output__"${folder////__}"".txt"
     nohup python3 ./pipeline_executable.py $folder_new $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx $nucleus_channel $cyto_channel $FISH_channel $FISH_second_channel $output_name "$path_to_config_file" >> $output_name &
     COUNTER=$((COUNTER+1))
     val1=$(($COUNTER%4)) 
     if [ $val1 -eq '0' ];then
     wait
     fi
done


#for folder in ${list_DUSP1_DEX[*]}; do
#     nohup python3 ./pipeline_executable.py  $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx $nucleus_channel $cyto_channel $FISH_channel $FISH_second_channel >> output.txt  &&
#     wait
#done
##########

# Deactivating the environment
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3
# ps -ef | grep python3 | grep "pipeline_"
# ps -ef | grep "python3 ./pipeline_executable.py *" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep "python3 ./pipeline_executable.py *" | awk '{print $2}')


# nvidia-smi | grep 'Default'
# top -u luisub

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm slurm* output__* temp_* *.tif

exit 0