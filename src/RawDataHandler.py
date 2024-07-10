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



from Utilities import Utilities


class RawDataHandler():
    """
    This class is to handle the raw data and get it to the experimental class, pipeline class and microscope class.

    It will take in the raw location of the data, on either the NAS or Local, and will download the data and convert to a standard format
    in a standard location.
    """
    def __init__(
                self,
                raw_data_location:str,
                local_or_NAS:str='NAS',
                dataformat:str='pycromanager',
                mask_location:str=None,
                share_name:str='share',
                connection_config_location:str=None,
                number_color_channels:int=None,
                number_of_fov:int=None,
                ) -> None:
        self.raw_data_location = raw_data_location
        self.local_or_NAS = local_or_NAS
        self.dataformat = dataformat
        self.mask_location = mask_location
        self.share_name = share_name
        self.connection_config_location = connection_config_location
        self.number_color_channels = number_color_channels
        self.number_of_fov = number_of_fov

        self.raw_data_location = pathlib.Path(raw_data_location)
        if self.mask_location is not None:
            self.mask_location = pathlib.Path(mask_location)
        

        if self.local_or_NAS == 'NAS':
            self.local_or_NAS = 1
            self.connection_config_location = pathlib.Path(connection_config_location)
        elif self.local_or_NAS == 'Local':
            self.local_or_NAS = 0

        util = Utilities()
        if dataformat == 'standard':
            self.local_data_dir, self.masks_dir, self.number_images, self.number_color_channels, self.list_files_names, self.list_images = util.read_images_from_folder( self.connection_config_location, 
                                                                                                                                                self.raw_data_location, 
                                                                                                                                                self.mask_location,  
                                                                                                                                                self.local_or_NAS)
        elif dataformat == 'pycromanager':
            self.local_data_dir, self.masks_dir, self.list_files_names, self.list_images_all_fov, self.list_images = util.convert_to_standard_format(data_folder_path=self.raw_data_location, 
                                                                                                                        path_to_config_file=self.connection_config_location, 
                                                                                                                        download_data_from_NAS = self.local_or_NAS,
                                                                                                                        use_metadata=1,
                                                                                                                        is_format_FOV_Z_Y_X_C=1)
        elif dataformat == 'other':
            if number_color_channels is None:
                raise ValueError('Please provide the number of color channels')
            if number_of_fov is None:
                raise ValueError('Please provide the number of fields of view')
            self.local_data_dir, self.masks_dir, self.list_files_names, self.list_images_all_fov, self.list_images = util.convert_to_standard_format(data_folder_path=self.raw_data_location, 
                                                                                                                        path_to_config_file=self.connection_config_location, 
                                                                                                                        download_data_from_NAS = self.local_or_NAS,
                                                                                                                        number_color_channels=number_color_channels,
                                                                                                                        number_of_fov=number_of_fov )
        else:
            raise ValueError('Data format not recognized. Please use either standard or pycromanager')
        
        self.successful_raw_data_input = self._check_if_import_was_successful()
        



    def _check_if_import_was_successful(self):
        # go to the local directory and check if the files are tif files
        for file in self.list_files_names:
            if not file.endswith('.tif'):
                return 0
        return 1
        




class PycromanagerDataHandler():
    pass







