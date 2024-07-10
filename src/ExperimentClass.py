from skimage import img_as_float64, img_as_uint
from skimage.filters import gaussian
from joblib import Parallel, delayed
import multiprocessing
import bigfish.stack as stack
import bigfish.plot as plot
import bigfish.detection as detection
import bigfish.multistack as multistack
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import re  
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.morphology import erosion
from scipy import ndimage
from scipy.optimize import curve_fit
import itertools
import glob
import tifffile
import sys
import datetime
import getpass
import pkg_resources
import platform
import math
from cellpose import models
import os; from os import listdir; from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl
mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import multiprocessing
from smb.SMBConnection import SMBConnection
import socket
import pathlib
import yaml
import shutil
from fpdf import FPDF
import gc
import pickle
import pycromanager as pycro

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch
    number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
    if number_gpus >1 : # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  str(np.random.randint(0,number_gpus,1)[0])        
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import zipfile
import seaborn as sns
import scipy.stats as stats
from  matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
font_props = {'size': 16}
import joypy
from matplotlib import cm
from scipy.ndimage import binary_dilation








class Experiment():
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image


    """
    def __init__(self, 
                initial_data_location:str,
                local_img_location:str,
                local_mask_location:str,
                number_of_images_to_process:int=None,
                number_of_channels:int=None,
                nucChannel:int=None,
                cytoChannel:int=None,
                FISHChannel:list=None,
                list_file_names:list=None,
                list_images:list=None
         ):
        self.initial_data_location = initial_data_location
        self.local_img_location = local_img_location
        self.local_mask_location = local_mask_location
        self.number_of_images_to_process = number_of_images_to_process
        self.number_of_channels = number_of_channels
        self.nucChannel = nucChannel
        self.cytoChannel = cytoChannel
        self.FISHChannel = FISHChannel
        self.list_file_names = list_file_names
        self.list_images = list_images

        self._check_if_local_img_location_exists()

        # if self.list_file_names is None:
        #     self.list_file_names = self._get_list_of_files(self.local_img_location)

        if self.number_of_images_to_process is None:
            self.number_of_images_to_process = len(self.list_file_names)
        
        if self.number_of_channels is None:
            # read the image to get the number of channels
            self.number_of_channels = self._get_number_of_channels()

        if self.FISHChannel is None:
            # return error if the FISHChannel is not defined
            raise ValueError('FISHChannel must be defined (why are you running this)')

    def _check_if_local_img_location_exists(self):
        if not (pathlib.Path(self.local_img_location).exists()):
            raise ValueError('Data has not been downloaded yet. Please download the data first')
        
    # def _get_list_of_files(self, list_dir:list):
    #     list_files = []
    #     for file in os.scandir(list_dir):
    #         list_files.append(file.filename)
    #     return list_files
    
    # def _get_number_of_channels(self):
    #     # read the first image to get the number of channels
    #     img = imread(self.list_images[0])
    #     return img.shape[-1]
    
    # def _check_if_intially_params_are_valid(self):
    #     # check if the initial parameters are valid
    #     # check if all parameters are defined
    #     if self.number_of_images_to_process is None:
    #         raise ValueError('number_of_images_to_process must be defined')
    #     if self.number_of_channels is None:
    #         raise ValueError('number_of_channels must be defined')
    #     if self.nucChannel is None:
    #         raise ValueError('nucChannel must be defined')
    #     if self.cytoChannel is None:
    #         raise ValueError('cytoChannel must be defined')
    #     if self.FISHChannel is None:
    #         raise ValueError('FISHChannel must be defined')
    #     if self.list_file_names is None:
    #         raise ValueError('list_file_names must be defined')
    #     if self.list_images is None:
    #         raise ValueError('list_images must be defined')
    #     if self.nas_or_local is None:
    #         raise ValueError('nas_or_local must be defined')
    #     if self.local_img_location is None:
    #         raise ValueError('local_img_location must be defined')
    #     if self.local_mask_location is None:
    #         raise ValueError('local_mask_location must be defined')
    #     if self.initial_data_location is None:
    #         raise ValueError('initial_data_location must be defined')
            

        
            
        

        
            



