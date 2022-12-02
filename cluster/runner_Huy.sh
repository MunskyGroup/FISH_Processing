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


#####################  Huys pape data 100X  ############################
<< ////
list_Huy=(\
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
path_to_config_file="$HOME/Desktop/config.yml"
NUMBER_OF_CORES=1
diameter_nucleus=180     # approximate nucleus size in pixels
diameter_cytosol=0       # approximate cytosol size in pixels
psf_z=350                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers
psf_yx=120               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers
voxel_size_z=500         # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=96        # Microscope conversion px to nanometers in the xy axis.
nucleus_channel='[0]'    # Channel to pass to python for nucleus segmentation
cyto_channel='None'      # Channel to pass to python for cytosol segmentation
FISH_channel='[1,2]'     # Channel to pass to python for spot detection
send_data_to_NAS=1       # If data sent back to NAS use 1
download_data_from_NAS=1
path_to_masks_dir='None' #'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0 # If true, it shows a all planes for the FISH plot detection. 
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
threshold_for_spot_detection=500 #'None'
save_filtered_images=0 # To save filtered images
////

####################  Huys pape data 60X  ############################
list_Huy=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_woStim' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_18minTPL_5uM' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_5hTPL_5uM' \
)

path_to_config_file="$HOME/Desktop/config.yml"
NUMBER_OF_CORES=1
diameter_nucleus=80     # approximate nucleus size in pixels
diameter_cytosol=0       # approximate cytosol size in pixels
psf_z=350                # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers
psf_yx=160               # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers
voxel_size_z=500         # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=160        # Microscope conversion px to nanometers in the xy axis.
nucleus_channel='[0]'    # Channel to pass to python for nucleus segmentation
cyto_channel='None'      # Channel to pass to python for cytosol segmentation
FISH_channel='[1,3]'     # Channel to pass to python for spot detection
send_data_to_NAS=0       # If data sent back to NAS use 1
download_data_from_NAS=1
path_to_masks_dir='None' #'Test/test_dir/masks_test_dir___nuc_120__cyto_220.zip'
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0 # If true, it shows a all planes for the FISH plot detection. 
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 
threshold_for_spot_detection='[300,400]'
save_filtered_images=0 # To save filtered images

# ########### PYTHON PROGRAM #############################
for folder in ${list_Huy[*]}; do
     output_names=""output__"${folder////__}"".txt"
     nohup python3 "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
     wait
done

# Deactivating the environment
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner_Huy.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3
# ps -ef | grep python3 | grep "pipeline_"
# ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}'  
# kill $(ps -ef | grep python3 | grep "pipeline_" | awk '{print $2}')

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif

exit 0