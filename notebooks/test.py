#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 04 03:01:24 2021

@author: luisub
"""

# Importing libraries
import sys
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import pathlib
import warnings
import shutil
import os
warnings.filterwarnings("ignore")

# Deffining directories
current_dir = pathlib.Path().absolute()
fa_dir = current_dir.parents[0].joinpath('src')

# Importing fish_analyses module
sys.path.append(str(fa_dir))
import fish_analyses as fa

#print(sys.argv)

## User passed arguments
remote_folder = sys.argv[1]
merge_images = int(sys.argv[2])
diamter_nucleus = int(sys.argv[3])                      # approximate nucleus size in pixels
diameter_cytosol = int(sys.argv[4])  #250                # approximate cytosol size in pixels
psf_z_1 = int(sys.argv[5])       #350                    # Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.
psf_yx_1 = int(sys.argv[6])      #150                    # Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.

remote_folder_path = pathlib.Path(remote_folder)

name_zip_file = (remote_folder_path.name +'__nuc_' + str(diamter_nucleus) +
                '__cyto_' + str(diameter_cytosol) +
                '__psfz_' + str(psf_z_1) +
                '__psfyx_' + str(psf_yx_1) )

print(name_zip_file)