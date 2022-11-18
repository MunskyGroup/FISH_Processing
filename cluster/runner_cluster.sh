#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=testing

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

list_test=(\
'Test/test_dir' \
'Test/test_dir1' \
) 

list_linda=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220728/RPE_GoldMedium_32P_COX-2_WOstm_Cyto543_560' \
'smFISH_images/Linda_smFISH_images/Confocal/20220728/RPE_GoldMedium_32P_COX-2_30ng_mL_IL-1B-1h_Cyto543_560' \
'smFISH_images/Linda_smFISH_images/Confocal/20220728/RPE_GoldMedium_32P_COX-2_30ng_mL_IL-1B-2h_Cyto543_560' \
'smFISH_images/Linda_smFISH_images/Confocal/20220801/RPE_GoldMedium_32P_COX-2_30ng_mL_IL-1B-4h_Cyto543_560' \
) 

list_linda2=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220914/A549_COX2_30ngmL_IL1B_2h_GoldMedia' \
'smFISH_images/Linda_smFISH_images/Confocal/20220914/A549_COX2_woSTM_GoldMedia' \
'smFISH_images/Linda_smFISH_images/Confocal/20220914/A549_NFKBIA_100nMDEX_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220914/A549_NFKBIA_woSTM' \
) 

list_A549_NFKBIA_test=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220927/A549_NFKBIA_woSTM' \
)

list_A549_NFKBIA=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220927/A549_NFKBIA_woSTM' \
'smFISH_images/Linda_smFISH_images/Confocal/20220927/A549_NFKBIA_10minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220927/A549_NFKBIA_20minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_30minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_40minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_50minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_60minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_75minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_90minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_120minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220928/A549_NFKBIA_150minDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220930/A549_NFKBIA_180minDEX' \
)

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 
NUMBER_OF_CORES=4

diameter_nucleus=90      # approximate nucleus size in pixels
diameter_cytosol=200     # approximate cytosol size in pixels
psf_z=350                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=120               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel='[0,0]'        # Channel to pass to python for nucleus segmentation
cyto_channel='[2,0]'           # Channel to pass to python for cytosol segmentation
FISH_channel='[1]'           # Channel to pass to python for spot detection
path_to_config_file="$HOME/FISH_Processing/config.yml"
send_data_to_NAS=1       # If data sent back to NAS use 1.
download_data_from_NAS=1
path_to_masks_dir='None' #'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0 # If true, it shows a all planes for the FISH plot detection. 
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
threshold_for_spot_detection=500 #'None'
# ########### PYTHON PROGRAM #############################
for folder in ${list_A549_NFKBIA[*]}; do
     output_names=""output__"${folder////__}"".txt"
     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES >> "$output_names" &
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
