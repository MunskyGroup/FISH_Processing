#!/bin/bash

# #####################################################################
# #####################################################################
# ########## Script used to process GR experiments  ###################
# #####################################################################
# #####################################################################
# ########### Authors: Luis Aguilera and Eric Ron -2024- ##############

# #####################################################################
# #####################################################################
# ########### ACTIVATE ENV ############################################
conda activate FISH_processing
export CUDA_VISIBLE_DEVICES=0,1

# #####################################################################
# #####################################################################
# ########### PATH TO IMAGES TO PROCESS #############################
# List of paths to images to process
GR_ICC_R1=(\
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_0min_050823' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_10min_050823' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_20min_050823' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_30min_050823' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_40min_050823' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_50min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_60min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_75min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_90min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_120min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_150min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_180min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_0min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_10min_050823' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_20min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_30min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_40min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_50min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_60min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_75min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_90min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_120min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_150min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_180min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_0min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_10min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_20min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_30min_050823' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_40min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_50min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_60min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_75min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_90min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_120min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_150min_050823' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_180min_050823' \
)
GR_ICC_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_0min_050823/masks_GR_ICC_3hr_R1_1nM_0min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_10min_050823/masks_GR_ICC_3hr_R1_1nM_10min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_20min_050823/masks_GR_ICC_3hr_R1_1nM_20min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_30min_050823/masks_GR_ICC_3hr_R1_1nM_30min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_40min_050823/masks_GR_ICC_3hr_R1_1nM_40min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/GR_ICC_3hr_R1_1nM_50min_050823/masks_GR_ICC_3hr_R1_1nM_50min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_60min_050823/masks_GR_ICC_3hr_R1_1nM_60min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_75min_050823/masks_GR_ICC_3hr_R1_1nM_75min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_90min_050823/masks_GR_ICC_3hr_R1_1nM_90min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_120min_050823/masks_GR_ICC_3hr_R1_1nM_120min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_150min_050823/masks_GR_ICC_3hr_R1_1nM_150min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_1nM_180min_050823/masks_GR_ICC_3hr_R1_1nM_180min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_0min_050823/masks_GR_ICC_3hr_R1_10nM_0min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_10min_050823/masks_GR_ICC_3hr_R1_10nM_10min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230516/GR_ICC_3hr_R1_10nM_20min_050823/masks_GR_ICC_3hr_R1_10nM_20min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_30min_050823/masks_GR_ICC_3hr_R1_10nM_30min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_40min_050823/masks_GR_ICC_3hr_R1_10nM_40min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_50min_050823/masks_GR_ICC_3hr_R1_10nM_50min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_60min_050823/masks_GR_ICC_3hr_R1_10nM_60min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_75min_050823/masks_GR_ICC_3hr_R1_10nM_75min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_90min_050823/masks_GR_ICC_3hr_R1_10nM_90min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_120min_050823/masks_GR_ICC_3hr_R1_10nM_120min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_150min_050823/masks_GR_ICC_3hr_R1_10nM_150min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_10nM_180min_050823/masks_GR_ICC_3hr_R1_10nM_180min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_0min_050823/masks_GR_ICC_3hr_R1_100nM_0min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_10min_050823/masks_GR_ICC_3hr_R1_100nM_10min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_20min_050823/masks_GR_ICC_3hr_R1_100nM_20min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_30min_050823/masks_GR_ICC_3hr_R1_100nM_30min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230522/GR_ICC_3hr_R1_100nM_40min_050823/masks_GR_ICC_3hr_R1_100nM_40min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_50min_050823/masks_GR_ICC_3hr_R1_100nM_50min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_60min_050823/masks_GR_ICC_3hr_R1_100nM_60min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_75min_050823/masks_GR_ICC_3hr_R1_100nM_75min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_90min_050823/masks_GR_ICC_3hr_R1_100nM_90min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_120min_050823/masks_GR_ICC_3hr_R1_100nM_120min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_150min_050823/masks_GR_ICC_3hr_R1_100nM_150min_050823___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/GR_ICC_3hr_R1_100nM_180min_050823/masks_GR_ICC_3hr_R1_100nM_180min_050823___nuc_100__cyto_200.zip' \
)
GR_ICC_R2=(\
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_0min_control_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_10min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_30min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_50min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_75min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_120min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_180min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_10min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_30min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_50min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_75min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_120min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_180min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_10min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_30min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_50min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_75min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_120min_062923_R2' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_180min_062923_R2' \
)
GR_ICC_R2_masks=(\
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_0min_control_062923_R2/masks_GR_ICC_0min_control_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_10min_062923_R2/masks_GR_ICC_1nM_10min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_30min_062923_R2/masks_GR_ICC_1nM_30min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_50min_062923_R2/masks_GR_ICC_1nM_50min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_75min_062923_R2/masks_GR_ICC_1nM_75min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_120min_062923_R2/masks_GR_ICC_1nM_120min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230707_GR_ICC_R2/GR_ICC_1nM_180min_062923_R2/masks_GR_ICC_1nM_180min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_10min_062923_R2/masks_GR_ICC_10nM_10min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_30min_062923_R2/masks_GR_ICC_10nM_30min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_50min_062923_R2/masks_GR_ICC_10nM_50min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_75min_062923_R2/masks_GR_ICC_10nM_75min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_120min_062923_R2/masks_GR_ICC_10nM_120min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_10nM_180min_062923_R2/masks_GR_ICC_10nM_180min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_10min_062923_R2/masks_GR_ICC_100nM_10min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_30min_062923_R2/masks_GR_ICC_100nM_30min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_50min_062923_R2/masks_GR_ICC_100nM_50min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_75min_062923_R2/masks_GR_ICC_100nM_75min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_120min_062923_R2/masks_GR_ICC_100nM_120min_062923_R2___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230713_GR_ICC_R2/GR_ICC_100nM_180min_062923_R2/masks_GR_ICC_100nM_180min_062923_R2___nuc_100__cyto_200.zip' \
)
GR_ICC_R3=(\
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_0min_control_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_10min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_30min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_50min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_75min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_120min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_180min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_10min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_30min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_50min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_75min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_10nM_120min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_10nM_180min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_10min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_30min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_50min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_75min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_120min_080823_R3' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_180min_080823_R3' \
)
GR_ICC_R3_masks=(\
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_0min_control_080823_R3/masks_GR_ICC_Dex_3hr_0min_control_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_10min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_10min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_30min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_30min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_50min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_50min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_75min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_75min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_120min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_120min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_1nM_180min_080823_R3/masks_GR_ICC_Dex_3hr_1nM_180min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_10min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_10min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_30min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_30min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_50min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_50min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230809_GR_ICC_R3/GR_ICC_Dex_3hr_10nM_75min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_75min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_10nM_120min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_120min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_10nM_180min_080823_R3/masks_GR_ICC_Dex_3hr_10nM_180min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_10min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_10min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_30min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_30min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_50min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_50min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_75min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_75min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_120min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_120min_080823_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230823/GR_ICC_Dex_3hr_100nM_180min_080823_R3/masks_GR_ICC_Dex_3hr_100nM_180min_080823_R3___nuc_100__cyto_200.zip' \
)



# #############################################################################
# #############################################################################
# ###########  FUNCTION TO RUN THE CODES #####################################
runnning_image_processing() {
    local -n arr1=$1 # The array of folders
    local -n arr2=$2 # The array of masks
    # #####################################################################
    # #####################################################################
    # ###################  CODE PARAMETERS ################################
    # Paths with configuration files
    path_to_executable="${PWD%/*/*/*}/src/pipeline_executable.py" 
    path_to_config_file="$HOME/Desktop/config.yml"
    NUMBER_OF_CORES=4
    diameter_nucleus=100                 # Approximate nucleus size in pixels
    diameter_cytosol=200                 # Approximate cytosol size in pixels
    psf_z=350                            # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
    psf_yx=160                           # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.
    voxel_size_z=500                     # Microscope conversion px to nanometers in the z axis.
    voxel_size_yx=160                    # Microscope conversion px to nanometers in the xy axis.
    channels_with_nucleus='[1]'          # Channel to pass to python for nucleus segmentation
    channels_with_cytosol='None'         # Channel to pass to python for cytosol segmentation
    channels_with_FISH='[0]'             # Channel to pass to python for spot detection
    send_data_to_NAS=1                   # If data sent back to NAS use 1.
    download_data_from_NAS=1             # Download data from NAS
    save_all_images=0                    # If true, it shows a all planes for the FISH plot detection.
    threshold_for_spot_detection='None'  # Thresholds for spot detection. Use an integer for a defined value, or 'None' for automatic detection.
    save_filtered_images=0               # Flag to save filtered images         
    optimization_segmentation_method='default' # optimization_segmentation_method = 'default' 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
    remove_z_slices_borders=0            # Use this flag to remove 2 z-slices from the top and bottom of the stack. This is needed to remove z-slices that are out of focus.
    remove_out_of_focus_images=0         # Flag to remove out of focus images
    save_pdf_report=0                    # Flag to save pdf report
    # Only modify the following three parameters if using terminator-scope.
    convert_to_standard_format=0         # Flag to convert to standard format. Only use if you want to convert the images to standard format.
    number_color_channels=0              # Number of color channels. Only use if you want to convert the images to standard format.
    number_of_fov=0                      # Number of fields of view. Only use if you want to convert the images to standard format.
    # #############################################################################
    # #############################################################################
    # ########### PYTHON PROGRAM USING DIR FOR MASKS #############################
    echo "Starting my job..."
    start_time=$(date +%s)
    for index in "${!arr1[@]}" ; do
        folder="${arr1[$index]}"
        output_names=""output__"${folder////__}"".txt"
        path_to_masks_dir="${arr2[$index]}"
        nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$channels_with_nucleus" "$channels_with_cytosol" "$channels_with_FISH" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images $remove_z_slices_borders $remove_out_of_focus_images $save_pdf_report $convert_to_standard_format $number_color_channels $number_of_fov >> "$output_names" &
        wait
    done
    end_time=$(date +%s)
    total_time=$(( (end_time - start_time) / 60 ))
    echo "Total time to complete the job: $total_time minutes"
}

# #############################################################################
# #############################################################################
# ########### Running the codes #########################

list_folders=("${GR_ICC_R1[@]}")
list_masks=("${GR_ICC_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${GR_ICC_R2[@]}")
list_masks=("${GR_ICC_R2_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${GR_ICC_R3[@]}")
list_masks=("${GR_ICC_R3_masks[@]}")
runnning_image_processing list_folders list_masks

# #############################################################################
# #############################################################################
# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source image_processing_script_GR.sh /dev/null 2>&1 & disown
# ########### TO MONITOR PROGRESS #########################
# ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')

exit 0

