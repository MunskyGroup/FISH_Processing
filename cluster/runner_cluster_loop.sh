#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=testing

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

####################  PATHS TO CODE FILES  ############################
path_to_config_file="$HOME/FISH_Processing/config.yml"
path_to_executable="${PWD%/*}/src/pipeline_executable.py" 

####################      REPOSITORIES    ############################
list_Huy=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_woStim' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_18minTPL_5uM' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_5hTPL_5uM' \
)

mask_list=(\
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_woStim/masks_MS2-CY5_Cyto543_560_woStim___nuc_70__cyto_0.zip' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_18minTPL_5uM/masks_MS2-CY5_Cyto543_560_18minTPL_5uM___nuc_70__cyto_0.zip' \
'smFISH_images/Linda_smFISH_images/Confocal/20220714/MS2-CY5_Cyto543_560_5hTPL_5uM/masks_MS2-CY5_Cyto543_560_5hTPL_5uM___nuc_70__cyto_0.zip' \
)

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

####################  CODE PARAMETERS ############################
NUMBER_OF_CORES=2
diameter_nucleus=71                        # Approximate nucleus size in pixels
diameter_cytosol=0                         # Approximate cytosol size in pixels
psf_z=350                                  # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers
psf_yx=160                                 # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers
voxel_size_z=500                           # Microscope conversion px to nanometers in the z axis.
voxel_size_yx=160                          # Microscope conversion px to nanometers in the xy axis.
nucleus_channel='[0]'                      # Channel to pass to python for nucleus segmentation
cyto_channel='None'                        # Channel to pass to python for cytosol segmentation
FISH_channel='[1,3]'                       # Channel to pass to python for spot detection
send_data_to_NAS=1                         # If data sent back to NAS use 1
download_data_from_NAS=1                   # If data downloaded from NAS use 1
optimization_segmentation_method='z_slice_segmentation' # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
save_all_images=0                          # If true, it shows a all planes for the FISH plot detection. 
save_filtered_images=0                     # To save filtered images

# ########### PYTHON PROGRAM #############################
#for folder in ${list_A549_NFKBIA[*]}; do
#     output_names=""output__"${folder////__}"".txt"
#     ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images $threshold_for_spot_detection $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
#     wait
#done

# List of thresholds to iterate
list_ts=('[400,450]' '[400,500]' '[400,550]' '[450,400]' '[450,500]' '[450,550]' '[500,400]' '[500,450]' '[500,550]' '[550,400]' '[550,450]' '[550,500]')

for threshold_for_spot_detection in ${list_ts[*]}; do
     counter=0
     for folder in ${list_Huy[*]}; do
          output_names=""output__"${folder////__}"".txt"
          path_to_masks_dir="${mask_list[counter]}"
          ~/.conda/envs/FISH_processing/bin/python "$path_to_executable" "$folder" $send_data_to_NAS $diameter_nucleus $diameter_cytosol $voxel_size_z $voxel_size_yx $psf_z $psf_yx "$nucleus_channel" "$cyto_channel" "$FISH_channel" "$output_names" "$path_to_config_file" $download_data_from_NAS $path_to_masks_dir $optimization_segmentation_method $save_all_images "$threshold_for_spot_detection" $NUMBER_OF_CORES $save_filtered_images >> "$output_names" &
          ((counter++))
          wait
     done
     wait
done


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
