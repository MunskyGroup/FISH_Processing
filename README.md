# FISH processing repository

Authors: Luis U. Aguilera, Joshua Cook, Brian Munsky

# Description

>:warning: **This software is in a very early and experimental stage**: it is intended to be used for testing and debugging purposes!

This repository uses [PySMB](https://github.com/miketeo/pysmb) to allow the user to transfer data between Network-attached storage (NAS) and remote or local server. Then it uses [Cellpose](https://github.com/MouseLand/cellpose) to detect and segment cells on microscope images. [Big-FISH](https://github.com/fish-quant/big-fish) is used to quantify the number of spots per cell. Data is processed using Pandas data frames for single-cell and cell population statistics.

## Code overview and architecture

<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/images/code_architecture.png alt="drawing" width="1200"/>

# Documentation accessible in readthedocs

The documentation is accessible in the following [link](https://fish-processing.readthedocs.io/en/latest/). 

# Colab implementation

 * Implementation in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1CQx4e5MQ0ZsZSQgqtLzVVh53dAg4uaQj?usp=sharing)

# Installation 

## Installation on a local computer

* To create a virtual environment, navigate to the location of the requirements file, and use:
```sh
 conda create -n FISH_processing python=3.8 -y
 source activate FISH_processing
```
* To install GPU for Cellpose (Optional step). Only for **Linux and Windows users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```sh
 conda install pytorch cudatoolkit=10.2 -c pytorch -y
```
* To install CPU for Cellpose (Optional step). Only for **Mac users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
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

If access is denied, please try using your Github token as follows:

```sh
git clone --depth 1 https://<token>@github.com/MunskyGroup/FISH_Processing.git
```
where `<token>` is your Github access token. 

* Move to the directory
```sh
cd FISH_Processing 
```
* Create an environment from this yml file.
```sh
/opt/ohpc/pub/apps/anaconda3/bin/conda env create -f FISH_env.yml
```

###  If an error occurs while creating the environment from the yml file, try using the following instructions.

* Create the environment
```sh
/opt/ohpc/pub/apps/anaconda3/bin/conda init 
/opt/ohpc/pub/apps/anaconda3/bin/conda create -n FISH_processing python=3.8 -y
/opt/ohpc/pub/apps/anaconda3/bin/conda init bash
source ~/.bashrc
```

* Then, activate the environment and manually install the dependencies.

```sh
/opt/ohpc/pub/apps/anaconda3/bin/conda activate FISH_processing
cd FISH_Processing
/opt/ohpc/pub/apps/anaconda3/bin/conda install pytorch cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

## Miscellaneous instructions:

To login to the NAS, it is needed to provide a configuration YML file with the format:
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

# Using this repository

Most codes are accessible as notebook scripts or executables. 

## To use the codes locally with an interactive environment, use the [notebooks folder](https://github.com/MunskyGroup/FISH_Processing/tree/main/notebooks)

- To process images use the notebook [FISH pipeline](https://github.com/MunskyGroup/FISH_Processing/blob/main/notebooks/FISH_pipeline.ipynb)

- After processing the images use the notebook [FISH pipeline](https://github.com/MunskyGroup/FISH_Processing/blob/main/notebooks/FISH_data_interpretaton.ipynb) to analyze multiple datasets 

## Executable codes are located in [cluster folder](https://github.com/MunskyGroup/FISH_Processing/tree/main/cluster)

- A [Bash script](https://github.com/MunskyGroup/FISH_Processing/blob/main/cluster/runner.sh) is used to execute a [python script](https://github.com/MunskyGroup/FISH_Processing/blob/main/cluster/pipeline_executable.py) containing the image processing pipeline. Please adapt these scripts to your specific configuration and target folders.

# Licenses for dependencies

Please check this [file](https://github.com/MunskyGroup/FISH_Processing/blob/main/Licenses_Dependencies.md) with the licenses for BIG-FISH, Cellpose, and PySMB.

# Citation

If you use this repository, make sure to cite BIG-FISH and Cellpose:

- [Big-FISH](https://github.com/fish-quant/big-fish):
Imbert, Arthur, et al. "FISH-quant v2: a scalable and modular analysis tool for smFISH image analysis." Biorxiv (2021).

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.
