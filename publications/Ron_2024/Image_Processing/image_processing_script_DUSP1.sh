#!/bin/bash

# #####################################################################
# #####################################################################
# ######### Script used to process DUSP1 experiments  #################
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

# DUSP1 Single Dex Concentration Time-Sweep
DUSP1_Dex_100nM_3hr_R1=(\
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_0min_20220224' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_10min_20220224' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_20min_20220224' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_30min_20220224' \
'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_40min_20220224' \
'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_50min_20220224' \
'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_60min_20220224' \
'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_75min_20220224' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_90min_20220224' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_120min_20220224' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_150min_20220224' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_180min_20220224' \
)
DUSP1_Dex_100nM_3hr_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_0min_20220224/masks_DUSP1_Dex_0min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_10min_20220224/masks_DUSP1_Dex_10min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_20min_20220224/masks_DUSP1_Dex_20min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_30min_20220224/masks_DUSP1_Dex_30min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_40min_20220224/masks_DUSP1_Dex_40min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_50min_20220224/masks_DUSP1_Dex_50min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_60min_20220224/masks_DUSP1_Dex_60min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_75min_20220224/masks_DUSP1_Dex_75min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_90min_20220224/masks_DUSP1_Dex_90min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_120min_20220224/masks_DUSP1_Dex_120min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_150min_20220224/masks_DUSP1_Dex_150min_20220224___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_180min_20220224/masks_DUSP1_Dex_180min_20220224___nuc_100__cyto_200.zip' \
)

DUSP1_Dex_100nM_3hr_R2=(\
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_0min' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_10min' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_20min' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_30min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_40min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_50min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_60min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_75min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_90min' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_120min' \
'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_150min' \
'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_180min' \
)
DUSP1_Dex_100nM_3hr_R2_masks=(\
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_0min/masks_DUSP1_100nM_Dex_R3_20220419_0min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_10min/masks_DUSP1_100nM_Dex_R3_20220419_10min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_20min/masks_DUSP1_100nM_Dex_R3_20220419_20min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_30min/masks_DUSP1_100nM_Dex_R3_20220419_30min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_40min/masks_DUSP1_100nM_Dex_R3_20220419_40min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_50min/masks_DUSP1_100nM_Dex_R3_20220419_50min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_60min/masks_DUSP1_100nM_Dex_R3_20220419_60min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_75min/masks_DUSP1_100nM_Dex_R3_20220419_75min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_90min/masks_DUSP1_100nM_Dex_R3_20220419_90min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_120min/masks_DUSP1_100nM_Dex_R3_20220419_120min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_150min/masks_DUSP1_100nM_Dex_R3_20220419_150min___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_180min/masks_DUSP1_100nM_Dex_R3_20220419_180min___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_100nM_3hr_R3=(\
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_0min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_10min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_20min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_30min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_40min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_50min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_60min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_75min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_90min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_120min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_150min_NoSpin_052722' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_180min_NoSpin_052722' \
)
DUSP1_Dex_100nM_3hr_R3_masks=(\
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_0min_NoSpin_052722/masks_DUSP1_Dex_100nM_0min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_10min_NoSpin_052722/masks_DUSP1_Dex_100nM_10min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_20min_NoSpin_052722/masks_DUSP1_Dex_100nM_20min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_30min_NoSpin_052722/masks_DUSP1_Dex_100nM_30min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_40min_NoSpin_052722/masks_DUSP1_Dex_100nM_40min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_50min_NoSpin_052722/masks_DUSP1_Dex_100nM_50min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_60min_NoSpin_052722/masks_DUSP1_Dex_100nM_60min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_75min_NoSpin_052722/masks_DUSP1_Dex_100nM_75min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_90min_NoSpin_052722/masks_DUSP1_Dex_100nM_90min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_120min_NoSpin_052722/masks_DUSP1_Dex_100nM_120min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_150min_NoSpin_052722/masks_DUSP1_Dex_100nM_150min_NoSpin_052722___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_180min_NoSpin_052722/masks_DUSP1_Dex_100nM_180min_NoSpin_052722___nuc_100__cyto_200.zip' \
)
# DUSP1 Single Time-Point Concentration Sweep
DUSP1_Dex_75min_conc_sweep_R1=(\
'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_0min_060322' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_1pM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_10pM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_100pM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1nM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_10nM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_100nM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1uM_75min_060322' \
'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_10uM_75min_060322' \
)
DUSP1_Dex_75min_conc_sweep_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_0min_060322/masks_DUSP1_conc_sweep_0min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_1pM_75min_060322/masks_DUSP1_conc_sweep_1pM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_10pM_75min_060322/masks_DUSP1_conc_sweep_10pM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_100pM_75min_060322/masks_DUSP1_conc_sweep_100pM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1nM_75min_060322/masks_DUSP1_conc_sweep_1nM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_10nM_75min_060322/masks_DUSP1_conc_sweep_10nM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_100nM_75min_060322/masks_DUSP1_conc_sweep_100nM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1uM_75min_060322/masks_DUSP1_conc_sweep_1uM_75min_060322___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_10uM_75min_060322/masks_DUSP1_conc_sweep_10uM_75min_060322___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_75min_conc_sweep_R2_masks=(
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_0min_071422/masks_DUSP1_conc_sweep_R2_0min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_1pM_75min_071422/masks_DUSP1_conc_sweep_R2_1pM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_10pM_75min_071422/masks_DUSP1_conc_sweep_R2_10pM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_100pM_75min_071422/masks_DUSP1_conc_sweep_R2_100pM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1nM_75min_071422/masks_DUSP1_conc_sweep_R2_1nM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10nM_75min_071422/masks_DUSP1_conc_sweep_R2_10nM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1uM_75min_071422/masks_DUSP1_conc_sweep_R2_1uM_75min_071422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10uM_75min_071422/masks_DUSP1_conc_sweep_R2_10uM_75min_071422___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_75min_conc_sweep_R2=(\
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_0min_071422' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_1pM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_10pM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_100pM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1nM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10nM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1uM_75min_071422' \
'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10uM_75min_071422' \
)
DUSP1_Dex_75min_conc_sweep_R3=(\
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_0nM_0min_Control_092022' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_1pM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_10pM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_100pM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_1nM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_10nM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_100nM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_1uM_75min_092022' \
'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_10uM_75min_092022' \
)
DUSP1_Dex_75min_conc_sweep_R3_masks=(\
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_0nM_0min_Control_092022/masks_DUSP1_Dex_Sweep_0nM_0min_Control_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_1pM_75min_092022/masks_DUSP1_Dex_Sweep_1pM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_10pM_75min_092022/masks_DUSP1_Dex_Sweep_10pM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_100pM_75min_092022/masks_DUSP1_Dex_Sweep_100pM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_100nM_75min_092022/masks_DUSP1_Dex_Sweep_100nM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_10nM_75min_092022/masks_DUSP1_Dex_Sweep_10nM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_1nM_75min_092022/masks_DUSP1_Dex_Sweep_1nM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_1uM_75min_092022/masks_DUSP1_Dex_Sweep_1uM_75min_092022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_10uM_75min_092022/masks_DUSP1_Dex_Sweep_10uM_75min_092022___nuc_100__cyto_200.zip' \
)
# DUSP1 Multiple Time-Points and Multiple Dex Concentrations
DUSP1_Dex_Conc_timesweep_R1=(\
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_0nM_Dex_0min_012623' \
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_30min_012623' \
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_50min_012623' \
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_75min_012623' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_90min_012623' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_120min_012623' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_180min_012623' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_1nM_Dex_30min_012623' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_50min_012623' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_75min_012623' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_90min_012623' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_120min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_1nM_Dex_180min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_30min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_50min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_75min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_90min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_120min_012623' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_180min_012623' \
)
DUSP1_Dex_Conc_timesweep_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_0nM_Dex_0min_012623/masks_DUSP1_0nM_Dex_0min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_30min_012623/masks_DUSP1_300pM_Dex_30min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_50min_012623/masks_DUSP1_300pM_Dex_50min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_90min_012623/masks_DUSP1_300pM_Dex_90min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_120min_012623/masks_DUSP1_300pM_Dex_120min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_180min_012623/masks_DUSP1_300pM_Dex_180min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230309/DUSP1_1nM_Dex_30min_012623/masks_DUSP1_1nM_Dex_30min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_50min_012623/masks_DUSP1_1nM_Dex_50min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_75min_012623/masks_DUSP1_1nM_Dex_75min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_90min_012623/masks_DUSP1_1nM_Dex_90min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_120min_012623/masks_DUSP1_1nM_Dex_120min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_1nM_Dex_180min_012623/masks_DUSP1_1nM_Dex_180min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_30min_012623/masks_DUSP1_10nM_Dex_30min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_50min_012623/masks_DUSP1_10nM_Dex_50min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_75min_012623/masks_DUSP1_10nM_Dex_75min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_90min_012623/masks_DUSP1_10nM_Dex_90min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_120min_012623/masks_DUSP1_10nM_Dex_120min_012623___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_180min_012623/masks_DUSP1_10nM_Dex_180min_012623___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_Conc_timesweep_R2=(\
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_0min_041223' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_30min_041223' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_50min_041223' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_75min_041223' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_90min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_120min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_180min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_30min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_50min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_75min_041223' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_90min_041223' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_120min_041223' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_180min_041223' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_30min_041223' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_50min_041223' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_75min_041223' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_90min_041223' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_120min_041223' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_180min_041223' \
)
DUSP1_Dex_Conc_timesweep_R2_masks=(\
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_0min_041223/masks_DUSP1_DexTimeConcSweep_0min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_30min_041223/masks_DUSP1_DexTimeConcSweep_300pM_30min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_50min_041223/masks_DUSP1_DexTimeConcSweep_300pM_50min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_75min_041223/masks_DUSP1_DexTimeConcSweep_300pM_75min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_90min_041223/masks_DUSP1_DexTimeConcSweep_300pM_90min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_120min_041223/masks_DUSP1_DexTimeConcSweep_300pM_120min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_180min_041223/masks_DUSP1_DexTimeConcSweep_300pM_180min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_30min_041223/masks_DUSP1_DexTimeConcSweep_1nM_30min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_50min_041223/masks_DUSP1_DexTimeConcSweep_1nM_50min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_75min_041223/masks_DUSP1_DexTimeConcSweep_1nM_75min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_90min_041223/masks_DUSP1_DexTimeConcSweep_1nM_90min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_120min_041223/masks_DUSP1_DexTimeConcSweep_1nM_120min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_180min_041223/masks_DUSP1_DexTimeConcSweep_1nM_180min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_30min_041223/masks_DUSP1_DexTimeConcSweep_10nM_30min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_50min_041223/masks_DUSP1_DexTimeConcSweep_10nM_50min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_75min_041223/masks_DUSP1_DexTimeConcSweep_10nM_75min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_90min_041223/masks_DUSP1_DexTimeConcSweep_10nM_90min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_120min_041223/masks_DUSP1_DexTimeConcSweep_10nM_120min_041223___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_180min_041223/masks_DUSP1_DexTimeConcSweep_10nM_180min_041223___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_Conc_timesweep_R3=(\
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_time_conc_sweep_0min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_30min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_50min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_75min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_90min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_120min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_180min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_30min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_50min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_90min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_120min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_180min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230614/DUSP1_Dex_10nM_30min_050223_R3_redo' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_50min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_75min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_90min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_120min_050223_R3' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_180min_050223_R3' \
)
DUSP1_Dex_Conc_timesweep_R3_masks=(\
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_time_conc_sweep_0min_050223_R3/masks_DUSP1_Dex_time_conc_sweep_0min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_30min_050223_R3/masks_DUSP1_Dex_300pM_30min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_50min_050223_R3/masks_DUSP1_Dex_300pM_50min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_75min_050223_R3/masks_DUSP1_Dex_300pM_75min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_90min_050223_R3/masks_DUSP1_Dex_300pM_90min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_120min_050223_R3/masks_DUSP1_Dex_300pM_120min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_180min_050223_R3/masks_DUSP1_Dex_300pM_180min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_30min_050223_R3/masks_DUSP1_Dex_1nM_30min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_50min_050223_R3/masks_DUSP1_Dex_1nM_50min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_90min_050223_R3/masks_DUSP1_Dex_1nM_90min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_120min_050223_R3/masks_DUSP1_Dex_1nM_120min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_180min_050223_R3/masks_DUSP1_Dex_1nM_180min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230614/DUSP1_Dex_10nM_30min_050223_R3_redo/masks_DUSP1_Dex_10nM_30min_050223_R3_redo___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_50min_050223_R3/masks_DUSP1_Dex_10nM_50min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_75min_050223_R3/masks_DUSP1_Dex_10nM_75min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_120min_050223_R3/masks_DUSP1_Dex_10nM_120min_050223_R3___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_180min_050223_R3/masks_DUSP1_Dex_10nM_180min_050223_R3___nuc_100__cyto_200.zip' \
)

# Control slides from unused 6hr single concentration experiments
DUSP1_Dex_100nM_6hr_R1=(\
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_0min_072022' \
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_150min_072022' \
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_180min_072022' \
)
DUSP1_Dex_100nM_6hr_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_0min_072022/masks_DUSP1_Dex_100nM_6hr_0min_072022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_150min_072022/masks_DUSP1_Dex_100nM_6hr_150min_072022___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_180min_072022/masks_DUSP1_Dex_100nM_6hr_180min_072022___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_100nM_6hr_R2=(\
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_0min_081822' \
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_30min_081822' \
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_60min_081822' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_90min_081822' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_120min_081822' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_150min_081822' \
'smFISH_images/Eric_smFISH_images/20220929/DUSP1_100nM_Dex_180min_081822' \
)
DUSP1_Dex_100nM_6hr_R2_masks=(\
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_0min_081822/masks_DUSP1_100nM_Dex_0min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_30min_081822/masks_DUSP1_100nM_Dex_30min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_60min_081822/masks_DUSP1_100nM_Dex_60min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_90min_081822/masks_DUSP1_100nM_Dex_90min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_120min_081822/masks_DUSP1_100nM_Dex_120min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_150min_081822/masks_DUSP1_100nM_Dex_150min_081822___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20220929/DUSP1_100nM_Dex_180min_081822/masks_DUSP1_100nM_Dex_180min_081822___nuc_100__cyto_200.zip' \
)
# DUSP1 100nM Dex and 5uM Triptolide experiments
DUSP1_Dex_100nM_TPL_R1=(\
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex_TPL_3hr_0min_Control_101422' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_Control_101422' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_10min_101422' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_30min_101422' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_60min_101422' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_Control_101422' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_10min_101422' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_30min_101422' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_60min_101422' \
)
DUSP1_Dex_100nM_TPL_R1_masks=(\
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex_TPL_3hr_0min_Control_101422/masks_DUSP1_Dex_TPL_3hr_0min_Control_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_Control_101422/masks_DUSP1_Dex100nM_75min_Control_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_10min_101422/masks_DUSP1_Dex100nM_75min_TPL5uM_10min_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_30min_101422/masks_DUSP1_Dex100nM_75min_TPL5uM_30min_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_60min_101422/masks_DUSP1_Dex100nM_75min_TPL5uM_60min_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_Control_101422/masks_DUSP1_Dex100nM_150min_Control_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_10min_101422/masks_DUSP1_Dex100nM_150min_TPL5uM_10min_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_30min_101422/masks_DUSP1_Dex100nM_150min_TPL5uM_30min_101422___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_60min_101422/masks_DUSP1_Dex100nM_150min_TPL5uM_60min_101422___nuc_100__cyto_200.zip' \
)
DUSP1_Dex_100nM_TPL_R2=(\
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_0min_110222' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_15min_110222' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_30min_110222' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_60min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_0min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_15min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_30min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_60min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_0min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_15min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_30min_110222' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_60min_110222' \
'smFISH_images/Eric_smFISH_images/20221109/DUSP1_Dex_180min_TPL_0min_110222' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_15min_110222' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_30min_110222' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_60min_110222' \
)
DUSP1_Dex_100nM_TPL_R2_masks=(\
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_0min_110222/masks_DUSP1_Dex_0min_TPL_0min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_15min_110222/masks_DUSP1_Dex_0min_TPL_15min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_30min_110222/masks_DUSP1_Dex_0min_TPL_30min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_60min_110222/masks_DUSP1_Dex_0min_TPL_60min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_0min_110222/masks_DUSP1_Dex_20min_TPL_0min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_15min_110222/masks_DUSP1_Dex_20min_TPL_15min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_30min_110222/masks_DUSP1_Dex_20min_TPL_30min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_60min_110222/masks_DUSP1_Dex_20min_TPL_60min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_0min_110222/masks_DUSP1_Dex_75min_TPL_0min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_15min_110222/masks_DUSP1_Dex_75min_TPL_15min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_30min_110222/masks_DUSP1_Dex_75min_TPL_30min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_60min_110222/masks_DUSP1_Dex_75min_TPL_60min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221109/DUSP1_Dex_180min_TPL_0min_110222/masks_DUSP1_Dex_180min_TPL_0min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_15min_110222/masks_DUSP1_Dex_180min_TPL_15min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_30min_110222/masks_DUSP1_Dex_180min_TPL_30min_110222___nuc_100__cyto_200.zip' \
'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_60min_110222/masks_DUSP1_Dex_180min_TPL_60min_110222___nuc_100__cyto_200.zip' \
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
    channels_with_nucleus='[2]'          # Channel to pass to python for nucleus segmentation
    channels_with_cytosol='[1]'          # Channel to pass to python for cytosol segmentation
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
list_folders=("${DUSP1_Dex_100nM_3hr_R1[@]}")
list_masks=("${DUSP1_Dex_100nM_3hr_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_100nM_3hr_R2[@]}")
list_masks=("${DUSP1_Dex_100nM_3hr_R2_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_100nM_3hr_R3[@]}")
list_masks=("${DUSP1_Dex_100nM_3hr_R3_masks[@]}")
runnning_image_processing list_folders list_masks

# DUSP1 Single Time-Point Concentration Sweep
list_folders=("${DUSP1_Dex_75min_conc_sweep_R1[@]}")
list_masks=("${DUSP1_Dex_75min_conc_sweep_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_75min_conc_sweep_R2[@]}")
list_masks=("${DUSP1_Dex_75min_conc_sweep_R2_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_75min_conc_sweep_R3[@]}")
list_masks=("${DUSP1_Dex_75min_conc_sweep_R3_masks[@]}")
runnning_image_processing list_folders list_masks

# DUSP1 Multiple Time-Points and Multiple Dex Concentrations
list_folders=("${DUSP1_Dex_Conc_timesweep_R1[@]}")
list_masks=("${DUSP1_Dex_Conc_timesweep_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_Conc_timesweep_R2[@]}")
list_masks=("${DUSP1_Dex_Conc_timesweep_R2_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_Conc_timesweep_R3[@]}")
list_masks=("${DUSP1_Dex_Conc_timesweep_R3_masks[@]}")
runnning_image_processing list_folders list_masks

# Control slides from unused 6hr single concentration experiments
list_folders=("${DUSP1_Dex_100nM_6hr_R1[@]}")
list_masks=("${DUSP1_Dex_100nM_6hr_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_100nM_6hr_R2[@]}")
list_masks=("${DUSP1_Dex_100nM_6hr_R2_masks[@]}")
runnning_image_processing list_folders list_masks

# DUSP1 100nM Dex and 5uM Triptolide experiments
list_folders=("${DUSP1_Dex_100nM_TPL_R1[@]}")
list_masks=("${DUSP1_Dex_100nM_TPL_R1_masks[@]}")
runnning_image_processing list_folders list_masks

list_folders=("${DUSP1_Dex_100nM_TPL_R2[@]}")
list_masks=("${DUSP1_Dex_100nM_TPL_R2_masks[@]}")
runnning_image_processing list_folders list_masks


# #############################################################################
# #############################################################################
# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source image_processing_script_DUSP1.sh /dev/null 2>&1 & disown
# ########### TO MONITOR PROGRESS #########################
# ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')

exit 0