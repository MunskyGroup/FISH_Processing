# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Python codes to analyze FISH images using (FISH-quant v2) BIG-FISH.
Created on Fri Oct 22 10:41:00 2021
Authors: Luis U. Aguilera, Joshua Cook, Brian Munsky.
'''

# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

import_libraries = 1
if import_libraries == 1:
    # importing Big-FISH
    import bigfish
    import bigfish.stack as stack
    import bigfish.segmentation as segmentation
    import bigfish.plot as plot
    import bigfish.detection as detection
    import pandas as pd
    # importing libraries
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt
    import re  
    import time
    from skimage.io import imread
    from scipy import ndimage
    import pysmb 
    # importing tensor flow
    import os; from os import listdir; from os.path import isfile, join
    # to remove tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


class NASConnection():
    '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass


class MergeChannels():
    '''
    This class takes images as arrays with format [Z,Y,X] and merge then in a numpy array with format [Z, Y, X, C].
    It recursively merges the channels in a new dimenssion in the array. Minimal number of Channels 2 maximum is 4
    
    Parameters
    --  --  --  --  -- 

    directory: str or PosixPath
        Directory containing the images to merge.
    substring_to_detect_in_file_name: str
        String with the prefix to detect in the files names. 
    save_figure: bool, optional
        Flag to save the merged images as .tif. The default is False. 
    '''
    def __init__(self, directory:str ,substring_to_detect_in_file_name:str = '.*_C0.tif', save_figure:bool = False ):
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
            self.directory = Path(directory)
        self.substring_to_detect_in_file_name = substring_to_detect_in_file_name
        self.save_figure=save_figure
    
    def merge(self):
        '''
        Method takes all the videos in the folder and merge those with similar names.

        Returns
        --  --  -- -
        list_file_names : List of strings 
            List with strings of names.
        list_merged_images : List of NumPy arrays. 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C].
        number_files : int. 
            Number of merged images in the folder.
        '''
        # This function 
        list_file_names =[]
        list_merged_images =[]  # list that stores all files belonging to the same image in a sublist
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        save_to_path = self.directory.joinpath('merged')
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file):
                    prefix = file.rpartition('_')[0]  # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif'))
                    list_file_names.append(prefix)
                    merged_img = np.concatenate([ imread(list_files_per_image[i])[..., np.newaxis] for i,_ in enumerate(list_files_per_image)],axis=-1).astype('uint16')
                    list_merged_images.append(merged_img) 
                    if self.save_figure ==1:
                        if not os.path.exists(str(save_to_path)):
                            os.makedirs(str(save_to_path))
                        tifffile.imsave(str(save_to_path.joinpath(prefix+'_merged'+'.tif')), merged_img, metadata={'axes': 'ZYXC'})
                    #del merged_img, list_files_per_image
        number_files = len(list_file_names)
        return list_file_names, list_merged_images, number_files,save_to_path


class CellSegmentation():
    '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass
    def nucleus_segmentation():
        pass
    def cytosol_segmentation():
        pass
    def automated_segmentation():
        pass


class SpotDetection():
    '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass


class DataProcessing():
    '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass


class Utilities ():
        '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass
    


