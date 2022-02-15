# FISH processing repository.

Authors: Luis U. Aguilera, Joshua Cook, Brian Munsky.

# Description

This repository uses [pysmb](https://github.com/miketeo/pysmb) to allow the user to transfer data between a Network-attached storage (NAS) and remote or local server. Then it uses [Cellpose](https://github.com/MouseLand/cellpose) to detect and segment cells on microscope images. [Big-FISH](https://github.com/fish-quant/big-fish) is used to quantify the number of spots per cell. Data is processed using Pandas data frames for single-cell and cell population statistics.

## Code overview and architecture

<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/code_architecture.png alt="drawing" width="1200"/>

# Colab implementation

 * Implementation in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1CQx4e5MQ0ZsZSQgqtLzVVh53dAg4uaQj?usp=sharing)

# Installation 

## Installation on a local computer

* To create a virtual environment, navigate to the location of the requirements file, and use:
```bash
 conda create -n FISH_processing python=3.8 -y
 source activate FISH_processing
```
* To install GPU for Cellpose (Optional step). Only for **Linux and Windows users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```
 conda install pytorch cudatoolkit=10.2 -c pytorch -y
```
* To install CPU for Cellpose (Optional step). Only for **Mac users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```
 conda install pytorch -c pytorch
```
* To include the rest of the requirements use:
```
 pip install -r requirements.txt
```

## Installation on the Keck-Cluster (Rocky Linux 8)

The following instructions are intended to use the codes on the Keck Cluster.

* Clone the repository to the cluster.
```
git clone --depth 1 https://<token>@github.com/MunskyGroup/FISH_Processing.git
```
* Move to the directory
```
cd FISH_Processing 
```
* Create an environment from this yml file.
```
/opt/ohpc/pub/apps/anaconda3/bin/conda env create -f FISH_env.yml
```

###  If an error occurs while creating the environment from the yml file, try using the following instructions.

* Create the environment
```
/opt/ohpc/pub/apps/anaconda3/bin/conda init 
/opt/ohpc/pub/apps/anaconda3/bin/conda create -n FISH_processing python=3.8 -y
/opt/ohpc/pub/apps/anaconda3/bin/conda init bash
source ~/.bashrc
```

* Then, activate the environment and manually install the dependencies.

```
/opt/ohpc/pub/apps/anaconda3/bin/conda activate FISH_processing
cd FISH_Processing
/opt/ohpc/pub/apps/anaconda3/bin/conda install pytorch cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

## Miscellaneous instructions:

Creating an environment file (yml) use:
```
conda env export > FISH_env.yml
```

Additional steps to deactivate or remove the environment from the computer:

* To deactivate the environment, use
```
 conda deactivate
```
* To remove the environment use:
```
 conda env remove -n FISH_processing
```

# Licenses for dependencies

## If you use this repository, make sure to cite:

- [Big-FISH](https://github.com/fish-quant/big-fish):
Imbert, Arthur, et al. "FISH-quant v2: a scalable and modular analysis tool for smFISH image analysis." Biorxiv (2021).

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.
