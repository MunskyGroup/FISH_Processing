# Image processing codes are used to quantify ​Fluorescence In Situ Hybridization (FISH) images.
Authors: Luis U. Aguilera, Eric Ron, Brian Munsky

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Description

This repository contains the codes used to quantify FISH images for the publication Ron et al. 2024. All microscope images are stored in the NAS (munsky-nas).  

## Folder overview

* [Image_Processing]: This folder contains two BASH scripts used to process all FISH images. 
  * image_processing_script_DUSP1.sh
  *  image_processing_script_GR.sh
  
* [Data_interpretation]: This folder contains ```Notebok_Data_Interpretation.ipynb```. This notebooks summarizes the complete dataframes for the GR and DUSP1 experiments. The code returns a summarized dataframe for all experimental conditions with the fields:
  *  Cell_id
  *  Condition 
  *  Replica 
  *  Dex_Conc 
  *  Time_index 
  *  Time_TPL 
  *  Nuc_Area 
  *  Cyto_Area 
  *  Nuc_GR_avg_int 
  *  Cyto_GR_avg_int
  *  Nuc_DUSP1_avg_int 
  *  Cyto_DUSP1_avg_int 
  *  RNA_DUSP1_nuc 
  *  RNA_DUSP1_cyto 

## Getting Started

To get started with these codes, clone this repository to your local machine. Ensure that you have the necessary software installed, including Bash and Jupyter Notebook. A complete description of how to use and install this library can be found here [FISH processing library](https://github.com/MunskyGroup/FISH_Processing)

## License

This project is licensed under the BSD 3-Clause License. For more details, see the [LICENSE](https://opensource.org/licenses/BSD-3-Clause) file.