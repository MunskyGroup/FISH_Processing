# ​Fluorescence In Situ Hybridization (FISH) - automated image processing

Authors: Luis U. Aguilera, Linda Forero-Quintero, Eric Ron, Joshua Cook, Brian Munsky

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1CQx4e5MQ0ZsZSQgqtLzVVh53dAg4uaQj?usp=sharing)
[![Documentation Status](https://readthedocs.org/projects/rsnaped/badge/?version=latest)](https://fish-processing.readthedocs.io/en/latest/)

# Description

Repository to automatically process ​Fluorescence In Situ Hybridization (FISH) images. This repository uses [PySMB](https://github.com/miketeo/pysmb) to allow the user to transfer data between Network-attached storage (NAS) and remote or local server. Then it uses [Cellpose](https://github.com/MouseLand/cellpose) to detect and segment cells on microscope images. [Big-FISH](https://github.com/fish-quant/big-fish) is used to quantify the number of spots per cell. Data is processed using Pandas data frames for single-cell and cell population statistics.

# Code architecture

<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/code_architecture.png alt="drawing" width="1200"/>

# Code overview

## Cell segmentation
    * The code can achieve accurate cell segmentation for the nucleus and cytosol in the images. The segmentation is performed using cellpose and complex optimization routines that ensure the maximum number of cells detected in the image.
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/cell_segmentation.png alt="drawing" width="1200"/>

## Spot detection
    * Spot detection is achieved using Big-FISH. Customization is added in this code to detect spots in multiple color channels. Additionally, this repository contains algorithms to measure spots that are co-detected in different color channels.  
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/single_cells.png alt="drawing" width="1200"/>

## Spot counting 
   * The code quantifies the number of spots per cell and allows the visualization of these numbers as a function of cell size.
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/scatter_cell_size_vs_spots.png alt="drawing" width="1200"/>

## Spot intensity quantification
    * The code allows quantifying the intensity of each spot, using the disk and a ring mask method developed by [Morisaki and Stasevich](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9191879/)
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/spot_intensities.png alt="drawing" width="1200"/>

## Data management
   * A complete data-frame for all processed images and cells is generated. This data-frame contains information about the location and intensity of each detected spot.
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/dataframe.png alt="drawing" width="1200"/>

## Data reproducibility report
    * To increase reproducibility a complete report is generated. This report contains information about the list of images processed, the specific parameters used to process the data, the user that processed the data, and the version of the modules and packages used.
<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/images_readme/metadata.png alt="drawing" width="1200"/>


# Installation 

## Installation on a local computer

* Clone the repository.
```sh
git clone --depth 1 https://github.com/MunskyGroup/FISH_Processing.git
```

* To create a virtual environment, navigate to the location of the requirements file, and use:
```sh
 conda create -n FISH_processing python=3.8 -y
 source activate FISH_processing
```
* To install pytorch for GPU usage in Cellpose (Optional step). Only for **Linux and Windows users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```sh
 conda install pytorch cudatoolkit=10.2 -c pytorch -y
```
* To install pytorch for CPU usage in Cellpose (Optional step). Only for **Mac users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```sh
 conda install pytorch -c pytorch
 
```
* To include the rest of the requirements use:
```sh
 pip install -r requirements.txt
```

## Installation on the Keck-Cluster (Rocky Linux 8)

The following instructions are intended to use the codes on the Keck Cluster.

* Clone the repository to the cluster.
```sh
git clone --depth 1 https://github.com/MunskyGroup/FISH_Processing.git
```

* Move to the directory
```sh
cd FISH_Processing 
```
* Create an environment from this yml file.
```sh
conda env create -f FISH_env.yml
```


# Using this repository

Most codes are accessible as notebook scripts or executables. 

### To use the codes locally with an interactive environment, use the [notebooks folder](https://github.com/MunskyGroup/FISH_Processing/tree/main/notebooks)

- To process images use the notebook [FISH pipeline](https://github.com/MunskyGroup/FISH_Processing/blob/main/notebooks/FISH_pipeline.ipynb)

- After processing the images use the notebook [FISH pipeline](https://github.com/MunskyGroup/FISH_Processing/blob/main/notebooks/FISH_data_interpretaton.ipynb) to analyze multiple datasets 

### Executable codes are located in [cluster folder](https://github.com/MunskyGroup/FISH_Processing/tree/main/cluster)

- A [Bash script](https://github.com/MunskyGroup/FISH_Processing/blob/main/cluster/runner.sh) is used to execute a [python script](https://github.com/MunskyGroup/FISH_Processing/blob/main/cluster/pipeline_executable.py) containing the image processing pipeline. Please adapt these scripts to your specific configuration and target folders.


# Miscellaneous instructions:

To login to the NAS, it is needed to provide a configuration YAML file with the format:
```yml
    user:
        username: user_name
        password: user_password
        remote_address : remote_ip_address
        domain: remote_domain
```

Creating an environment file (yml) use:
```sh
conda env export > FISH_env.yml
```

Additional steps to deactivate or remove the environment from the computer:

* To deactivate the environment, use
```sh
 conda deactivate
```
* To remove the environment use:
```sh
 conda env remove -n FISH_processing
```

To create the documentation use the following modules.
```sh
pip install sphinx
pip install sphinx_rtd_theme
pip install Pygments
```

# Licenses for dependencies

Please check this [file](https://github.com/MunskyGroup/FISH_Processing/blob/main/Licenses_Dependencies.md) with the licenses for BIG-FISH, Cellpose, and PySMB.


# Citation

If you use this repository, make sure to cite BIG-FISH and Cellpose:

- [Big-FISH](https://github.com/fish-quant/big-fish):
Imbert, Arthur, et al. "FISH-quant v2: a scalable and modular tool for smFISH image analysis." RNA (2022): rna-079073.

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.
