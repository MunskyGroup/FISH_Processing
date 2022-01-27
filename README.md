# FISH_Processing with (FISH-quant v2) Big-FISH and Cellpose.

Authors: Luis U. Aguilera, Joshua Cook, Brian Munsky.

## Description

This library is intended to analyze sm-FISH images using (FISH-quant v2) [Big-FISH](https://github.com/fish-quant/big-fish) and [Cellpose](https://github.com/MouseLand/cellpose).

## Code architecture

<img src= https://github.com/MunskyGroup/FISH_Processing/raw/main/docs/code_architecture.png alt="drawing" width="1200"/>

## Colab implementation

 * Implementation in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1CQx4e5MQ0ZsZSQgqtLzVVh53dAg4uaQj?usp=sharing)


## Installation

* To create a virtual environment navigate to the location of the requirements file, and use:
```bash
    conda create -n FISH_processing python=3.6 -y
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
* To include the rest of requirements use:
```
    pip install -r requirements.txt --no-cache-dir
```
Additional steps to deactivate or remove the environment from the computer:
* To deactivate the environment use
```
    conda deactivate
```
* To remove the environment use:
```
    conda env remove -n FISH_processing
```

## References for dependencies

- [Big-FISH](https://github.com/fish-quant/big-fish):
Imbert, Arthur, et al. "FISH-quant v2: a scalable and modular analysis tool for smFISH image analysis." Biorxiv (2021).

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.

## Licenses for dependencies

- License for [Big-FISH](https://github.com/fish-quant/big-fish): BSD 3-Clause License. Copyright © 2020, Arthur Imbert. All rights reserved.

- License for [Cellpose](https://github.com/MouseLand/cellpose): BSD 3-Clause Copyright © 2020 Howard Hughes Medical Institute. 
