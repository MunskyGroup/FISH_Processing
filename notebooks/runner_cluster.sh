#!/bin/sh
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

# LIST OF FOLDERS
list_DUSP1_DEX=(\
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_0min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_10min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_20min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_30min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_40min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_50min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_60min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_75min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_90min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_120min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_150min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_180min' \
)

list_test=(\
'Test/test_dir' \
'Test/test_dir1' \
) 

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

send_data_to_NAS=1       # If data sent back to NAS use 1.
diamter_nucleus=100      # approximate nucleus size in pixels
diameter_cytosol=250     # approximate cytosol size in pixels
psf_z=300                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=105               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel=0        # Channel to pass to python for nucleus segmentation
cyto_channel=2           # Channel to pass to python for cytosol segmentation
FISH_channel=1           # Channel to pass to python for spot detection
FISH_second_channel=0    # Channel to pass to python for spot detection in a second Channel, if 0 is ignored.

# ########### PYTHON PROGRAM #############################
#COUNTER=0
for folder in ${list_test[*]}; do
     output_names=""output__"${folder////__}"".txt"
     ~/.conda/envs/FISH_processing/bin/python ./pipeline_executable.py $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx $nucleus_channel $cyto_channel $FISH_channel $FISH_second_channel $output_names >> $output_names &
     wait
done

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster_new.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm slurm* out* temp_* 

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue#!/bin/sh
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

# LIST OF FOLDERS
list_DUSP1_DEX=(\
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_0min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_10min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_20min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_30min' \
'smFISH_images/Eric_smFISH_images/20220126/DUSP1_Dex_40min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_50min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_60min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_75min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_90min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_120min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_150min' \
'smFISH_images/Eric_smFISH_images/20220131/DUSP1_Dex_180min' \
)

list_test=(\
'Test/test_dir' \
'Test/test_dir1' \
) 

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

send_data_to_NAS=1       # If data sent back to NAS use 1.
diamter_nucleus=100      # approximate nucleus size in pixels
diameter_cytosol=250     # approximate cytosol size in pixels
psf_z=300                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=105               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel=0        # Channel to pass to python for nucleus segmentation
cyto_channel=2           # Channel to pass to python for cytosol segmentation
FISH_channel=1           # Channel to pass to python for spot detection
FISH_second_channel=0    # Channel to pass to python for spot detection in a second Channel, if 0 is ignored.

# ########### PYTHON PROGRAM #############################
#COUNTER=0
for folder in ${list_test[*]}; do
     output_names=""output__"${folder////__}"".txt"
     ~/.conda/envs/FISH_processing/bin/python ./pipeline_executable.py $folder $send_data_to_NAS $diamter_nucleus $diameter_cytosol $psf_z $psf_yx $nucleus_channel $cyto_channel $FISH_channel $FISH_second_channel $output_names >> $output_names &
     wait
done

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster_new.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* error_file*

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue