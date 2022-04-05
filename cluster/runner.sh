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

# To enable GPUs on code
# https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on

# Declare a string array

list_test_b=(\
'Test/test_dir' \
) 


diameter_nucleus=110               # approximate nucleus size in pixels
diameter_cytosol=250               # approximate cytosol size in pixels
psf_z=350                          # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=120                         # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
cyto_channel='[0,1,2]'           # Channel to pass to python for cytosol segmentation
FISH_channel='[1]'           # Channel to pass to python for spot detection
threshold_for_spot_detection=300
list_cox_il=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_wo_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_6h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_6h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_6h' \
)

#diameter_nucleus=120               # approximate nucleus size in pixels
#diameter_cytosol=220               # approximate cytosol size in pixels
#psf_z=300                          # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
#psf_yx=140                         # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
#nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
#cyto_channel='[2]'           # Channel to pass to python for cytosol segmentation
#FISH_channel='[1]'           # Channel to pass to python for spot detection
list_cox_dex=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220324/GAPDH-Cy3_COX-2-Cy5_WO_Dex_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220324/GAPDH-Cy3_COX-2-Cy5_10min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220325/GAPDH-Cy3_COX-2-Cy5_20min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220325/GAPDH-Cy3_COX-2-Cy5_30min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_45min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_60min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_2h_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_4h_100nM_DEX' \
)

#diameter_nucleus=100                # approximate nucleus size in pixels
#diameter_cytosol=180                # approximate cytosol size in pixels
#psf_z=300                           # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
#psf_yx=140                          # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
#nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
#cyto_channel='[2]'           # Channel to pass to python for cytosol segmentation
#FISH_channel='[1]'           # Channel to pass to python for spot detection
list_cox_il_tp=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220324/GAPDH-Cy3_COX-2-Cy5_WO_Dex_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_6min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_12min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220329/GAPDH-Cy3_COX-2-Cy5_18min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220329/GAPDH-Cy3_COX-2-Cy5_24min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220329/GAPDH-Cy3_COX-2-Cy5_30min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220330/GAPDH-Cy3_COX-2-Cy5_36min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220330/GAPDH-Cy3_COX-2-Cy5_42min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220331/GAPDH-Cy3_COX-2-Cy5_48min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220331/GAPDH-Cy3_COX-2-Cy5_60min_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_2h_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220331/GAPDH-Cy3_COX-2-Cy5_3h_30ng_mL_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220401/GAPDH-Cy3_COX-2-Cy5_4h_30ng_mL_IL1B' \
)


#diameter_nucleus=120      # approximate nucleus size in pixels
#diameter_cytosol=220     # approximate cytosol size in pixels
#psf_z=300                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
#psf_yx=150               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
#nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
#cyto_channel='[2]'           # Channel to pass to python for cytosol segmentation
#FISH_channel='[1]'           # Channel to pass to python for spot detection
path_to_config_file="$HOME/Desktop/config.yml"
send_data_to_NAS=1       # If data sent back to NAS use 1.
download_data_from_NAS=1 # If data is downloaded from NAS use 1
path_to_masks_dir='None' #'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0 # If true, it shows a all planes for the FISH plot detection. 
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
#########for loop
# over different parameters above
# pick ones with most cells, 3-4 for for these folders
#look for which has what effect on the output files (spot detection etc.)

# var_new="${var//\\//}"
# // replace every
# \\ backslash
# / with
# / slash
maximum_parallel_iterations=3
# ########### PYTHON PROGRAM #############################
COUNTER=0
for folder in ${list_cox_il[*]}; do
     output_names=""output__"${folder////__}"".txt"
     nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection >> "$output_names" &
     COUNTER=$((COUNTER+1))
     val1=$(($COUNTER%maximum_parallel_iterations)) 
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
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')


# nvidia-smi | grep 'Default'
# top -u luisub | grep 'python3'

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif

exit 0