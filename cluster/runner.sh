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

#list_test_b=(\
#'Test/test_dir' \
#) 

# 'smFISH_images/Linda_smFISH_images/Confocal/20220927/A549_NFKBIA_woSTM' \

list_Huys_paper_test=(\
'smFISH_images/Linda_smFISH_images/Confocal/20211014/MS2-CY5-0minTPL' \
)

list_Huys_paper=(\
'smFISH_images/Linda_smFISH_images/Confocal/20211014/MS2-CY5-0minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211014/MS2-CY5-3minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211015/MS2-CY5-6minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211015/MS2-CY5-9minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211015/MS2-CY5-12minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211019/MS2-CY5-15minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211019/MS2-CY5-18minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211019/MS2-CY5-21minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211021/MS2-CY5-24minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211021/MS2-CY5-27minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211021/MS2-CY5-30minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20211021/MS2-CY5-60minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20210921/MS2-Cy5-120minTPL' \
'smFISH_images/Linda_smFISH_images/Confocal/20210921/MS2-Cy5-240minTPL' \
)

list_A549_NFKBIA_test=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220930/A549_NFKBIA_180minDEX' \
)

list_A549_NFKB_AB_488_IKBA_short=(\
'smFISH_images/Linda_smFISH_images/Confocal/20221101/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_60min_DEX_60min_C2' \
)

list_A549_NFKB_AB_488_COX2=(\
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_DEX_60min_B4' \
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_DEX_120min_C4' \
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_woDEX_A4' \
)

list_A549_NFKB_AB_488_IKBA=(\
'smFISH_images/Linda_smFISH_images/Confocal/20221027/A549_NFKB_AB_488_IKBA-Cy5_woSTM' \
'smFISH_images/Linda_smFISH_images/Confocal/20221027/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_60min_woDEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20221031/A549_NFKB_AB_488_IKBA-Cy5_DEX_60min_woIL-1B_A2' \
'smFISH_images/Linda_smFISH_images/Confocal/20221031/A549_NFKB_AB_488_IKBA-Cy5_DEX_120min_woIL-1B_A3' \
'smFISH_images/Linda_smFISH_images/Confocal/20221031/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_20min_DEX_60min_B2' \
'smFISH_images/Linda_smFISH_images/Confocal/20221031/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_20min_woDEX_B1' \
'smFISH_images/Linda_smFISH_images/Confocal/20221101/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_20min_DEX_120min_B3' \
'smFISH_images/Linda_smFISH_images/Confocal/20221101/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_60min_DEX_60min_C2' \
'smFISH_images/Linda_smFISH_images/Confocal/20221101/A549_NFKB_AB_488_IKBA-Cy5_IL-1B_60min_DEX_120min_C3' \
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

list_dirs=(\
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_woDEX_A4' \
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_DEX_60min_B4' \
'smFISH_images/Linda_smFISH_images/Confocal/20221102/A549_NFKB_AB_488_COX2-Cy5_IL-1B_60min_DEX_120min_C4' \
)

list_test_b=(\
'smFISH_images/Eric_smFISH_images/20211109/MS2_Cy3_TPL_0min' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_woDex' \
)



list_Dex_R1=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_woDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220121/GAPDH-Cy3_NFKBIA-Cy5_5min_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220121/GAPDH-Cy3_NFKBIA-Cy5_10min_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_15min_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220124/GAPDH-Cy3_NFKBIA-Cy5_30min_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220117/GAPDH-Cy3_NFKBIA-Cy5_1h_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220114/GAPDH-Cy3_NFKBIA-Cy5_2h_100nMDex' \
'smFISH_images/Linda_smFISH_images/Confocal/20220117/GAPDH-Cy3_NFKBIA-Cy5_4h_100nMDex' \
)

list_Dex_R2=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_WO_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220216/GAPDH-Cy3_NFKBIA-Cy5_5min_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220216/GAPDH-Cy3_NFKBIA-Cy5_10min_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220216/GAPDH-Cy3_NFKBIA-Cy5_15min_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220215/GAPDH-Cy3_NFKBIA-Cy5_30min_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_1h_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_2h_100nM_DEX_Rep2' \
'smFISH_images/Linda_smFISH_images/Confocal/20220214/GAPDH-Cy3_NFKBIA-Cy5_4h_100nM_DEX_Rep2' )

#diameter_nucleus=90               # approximate nucleus size in pixels
#diameter_cytosol=140               # approximate cytosol size in pixels
#psf_z=350                          # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
#psf_yx=120                         # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
#nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
#cyto_channel='[0,2]'           # Channel to pass to python for cytosol segmentation
#FISH_channel='[1]'           # Channel to pass to python for spot detection
#threshold_for_spot_detection=400#
#list_cox_il=(\
#'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_wo_IL-1B' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_2h' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_6h' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_2h' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_6h' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_2h' \
#'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_6h' \
#)

<< ////
diameter_nucleus=90               # approximate nucleus size in pixels
diameter_cytosol=180               # approximate cytosol size in pixels
psf_z=350                          # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=120                         # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
cyto_channel='[0,2]'           # Channel to pass to python for cytosol segmentation
FISH_channel='[1]'           # Channel to pass to python for spot detection
threshold_for_spot_detection='None'
list_cox_dex=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220324/GAPDH-Cy3_COX-2-Cy5_WO_Dex_IL1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220324/GAPDH-Cy3_COX-2-Cy5_10min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220325/GAPDH-Cy3_COX-2-Cy5_20min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220325/GAPDH-Cy3_COX-2-Cy5_30min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_45min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_60min_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_2h_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220328/GAPDH-Cy3_COX-2-Cy5_4h_100nM_DEX' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_wo_IL-1B' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_30ng_ml_IL-1B_6h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220314/GAPDH-Cy3_COX-2-Cy5_100ng_ml_IL-1B_6h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_2h' \
'smFISH_images/Linda_smFISH_images/Confocal/20220315/GAPDH-Cy3_COX-2-Cy5_300ng_ml_IL-1B_6h' \
)
////




<< ////
list_cox_il=(\
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
////

path_to_config_file="$HOME/Desktop/config.yml"
NUMBER_OF_CORES=1
diameter_nucleus=91      # approximate nucleus size in pixels
diameter_cytosol=201     # approximate cytosol size in pixels
psf_z=350                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx=160               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
voxel_size_z=500    #300                      # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=160 #103   #103                 # Microscope conversion px to nanometers in the xy axis.
nucleus_channel='[0]'        # Channel to pass to python for nucleus segmentation
cyto_channel='[1,2]'           # Channel to pass to python for cytosol segmentation
FISH_channel='[3]'           # Channel to pass to python for spot detection
send_data_to_NAS=1       # If data sent back to NAS use 1.
download_data_from_NAS=1
path_to_masks_dir='None' #'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0 # If true, it shows a all planes for the FISH plot detection. 
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
threshold_for_spot_detection=1500 #'None'
save_filtered_images=1


#########for loop
# over different parameters above
# pick ones with most cells, 3-4 for for these folders
#look for which has what effect on the output files (spot detection etc.)

# var_new="${var//\\//}"
# // replace every
# \\ backslash
# / with
# / slash
# ########### PYTHON PROGRAM #############################
for folder in ${list_dirs[*]}; do
     output_names=""output__"${folder////__}"".txt"
     nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
     wait
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
# ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')


# nvidia-smi | grep 'Default'

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif

exit 0