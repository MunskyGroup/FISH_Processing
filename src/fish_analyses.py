# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module uses `pysmb <https://github.com/miketeo/pysmb>`_ to allow the user to transfer data between a Network-attached storage (NAS) and remote or local server. Then it uses `Cellpose <https://github.com/MouseLand/cellpose>`_ to detect and segment cells on microscope images. `Big-FISH <https://github.com/fish-quant/big-fish>`_ is used to quantify the number of spots per cell. Data is processed using Pandas data frames for single-cell and cell population statistics.
Authors: Luis U. Aguilera, Joshua Cook, and Brian Munsky.

If you use this repository, make sure to cite:

`Big-FISH <https://github.com/fish-quant/big-fish>`_ :
Imbert, Arthur, et al. "FISH-quant v2: a scalable and modular tool for smFISH image analysis." RNA (2022): rna-079073.

`Cellpose <https://github.com/MouseLand/cellpose>`_ :
Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.

'''

# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

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
from scipy import ndimage
import glob
import tifffile
import sys
import datetime
import getpass
import pkg_resources
import platform
from cellpose import models
import os; from os import listdir; from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from skimage.io import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
plt.style.use('ggplot')  # ggplot  #default
from joblib import Parallel, delayed
import multiprocessing
from smb.SMBConnection import SMBConnection
import socket
import pathlib
import yaml
import shutil
from fpdf import FPDF
import gc
# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch
    number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
    if number_gpus >1 : # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  str(np.random.randint(0,number_gpus,1)[0])        
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')


class Banner():
    def __init__(self):
        '''
        This class prints a banner with a fish and the name of the authors of this repository. 
        '''
        pass
        
    def print_banner(self):
        '''
        This method prints the banner as text output. No parameters are needed.
        '''
        print(" \n"
            "FISH processing repository by : \n"
            "Luis U. Aguilera, Joshua Cook, Tim Stasevich, and Brian Munsky. \n" 
            " ____________________________________________________________  \n"      
            "|                      ,#^^^^^^^%&&&                         | \n"
            "|  .&.                 &.           ,&&&___                  | \n"
            "|  &  &         ___&&&/                    (&&&&____         | \n"
            "|  &    &,____#&                   .       #.       %&**,    | \n"
            "|  /(                  &         ,%       &       %     ,&   | \n"
            "|    &          &.                       %.      %&%     &*  | \n"
            "|     &&         *         .%            &             &(    | \n"
            "|   &                &(           ,#     .%             ,.&  | \n"
            "|  &    _&&__#&.     &&           &.      ,&         ,%&     | \n"
            "|  &  (%        #&,___                      (-***%&%^        | \n"
            "|  & &                %&&&(,.      .*#&&&&&%.                | \n"
            "|                          &    ,%%%%                        | \n"
            "|___________________________/%%^_____________________________| \n" )
        return None


class Utilities():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass
    
    # This function is intended to merge masks in a single image
    def merge_masks (self,list_masks):
        '''
        This method is intended to merge a list of images into a single image (Numpy array) where each cell is represented by an integer value.
        
        Parameters
        
        list_masks : List of Numpy arrays.
            List of Numpy arrays, where each array has dimensions [Y, X] with values 0 and 1, where 0 represents the background and 1 the cell mask in the image.
        '''
        n_masks = len(list_masks)
        print('n_masks',n_masks)
        if not ( n_masks is None):
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                base_image = np.zeros_like(list_masks[0])
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    tested_mask = np.where(list_masks[nm-1] == 1, nm, 0)
                    base_image = base_image + tested_mask
            # making zeros all elements outside each mask, and once all elements inside of each mask.
            else:  # do nothing if only a single mask is detected per image.
                base_image = list_masks[0]
        else:
            base_image =[]
        masks = base_image.astype(np.uint8)
        return masks
    
    def separate_masks (self,masks):
        '''
        This method is intended to separate an image (Numpy array) with multiple masks into a list of Numpy arrays where each cell is represented individually in a new NumPy array.
        
        Parameters
        
        masks : Numpy array.
            Numpy array with dimensions [Y, X] with values from 0 to n where n is the number of masks in the image.
        '''
        list_masks = []
        n_masks = np.max(masks)
        if not ( n_masks is None):
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    mask_copy = masks.copy()
                    tested_mask = np.where(mask_copy == nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                    list_masks.append(tested_mask)
            else:  # do nothing if only a single mask is detected per image.
                list_masks.append(masks)
        else:
            list_masks.append(masks)
        return list_masks
    
    def convert_to_int8(self,image,rescale=True,min_percentile=1, max_percentile=98):
        '''
        This method converts images from int16 to uint8. Optionally, the image can be rescaled and stretched.
        
        Parameters
        
        image : NumPy array
            NumPy array with dimensions [Y, X, C]. The code expects 3 channels (RGB). If less than 3 values are passed, the array is padded with zeros.
        rescale : bool, optional
            If True it rescales the image to stretch intensity values to a 95 percentile, and then rescale the min and max intensity to 0 and 255. The default is True. 
        '''
        if rescale == True:
            image = RemoveExtrema(image,min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers() 
            #image = stack.rescale(image, channel_to_stretch=0, stretching_percentile=99)
        #if rescale == True:
        #    imin, imax = np.min(image), np.max(image) 
        #    image -= imin
        #    image_float = np.array(image, 'float32')
        #    image_float *= 255./(imax-imin)
        #    image_new = np.array(np.round(image_float), 'uint8')
        #else:
        image_new= np.zeros_like(image)
        for i in range(0, image.shape[2]):  # iterate for each channel
            image_new[:,:,i]= (image[:,:,i]/ image[:,:,i].max()) *255
            image_new = np.uint8(image_new)
        # padding with zeros the channel dimension.
        while image_new.shape[2]<3:
            zeros_plane = np.zeros_like(image_new[:,:,0])
            image_new = np.concatenate((image_new,zeros_plane[:,:,np.newaxis]),axis=2)
        return image_new


class NASConnection():
    '''
    This class is intended to establish a connection between Network-Attached storage and a remote (or local) computer using `pysmb <https://github.com/miketeo/pysmb>`_ . The class allows the user to connect to NAS, download specific files, and write backfiles to NAS.
    This class doesn't allow the user to delete, modify or overwrite files in NAS. For a complete description of pysmb check the `pysmb documentation <https://pysmb.readthedocs.io/en/latest/>`_ .
    To use this class, you need to:
    
    1) Use the university's network or use the two-factor authentication to connect to the university's VPN.
    2) You need to create a configuration YAML file with the following format:
    
    .. code-block:: bash

        user:
        username: name_of_the_user_in_the_nas_server
        password: user_password_in_the_nas_server 
        remote_address : ip or name for the nas server
        domain: domain for the nas server 
    
    Parameters
    
    path_to_config_file : str, or Pathlib object
        The path in the local computer contains the config file.
    share_name: str
        Name of the share partition to access in NAS. The default is 'share'.
    '''
    def __init__(self,path_to_config_file,share_name = 'share'):
        # Loading credentials
        conf = yaml.safe_load(open(str(path_to_config_file)))
        usr = conf['user']['username']
        pwd = conf['user']['password']
        remote_address = conf['user']['remote_address']
        domain = conf['user']['domain']
        # LOCAL NAME
        local_name = socket.gethostbyname(socket.gethostname())
        # SERVER NAME
        self.share_name = share_name
        self.server_name, _, _ = socket.gethostbyaddr(remote_address)
        # Defining the connection to NAS
        self.conn = SMBConnection(username=usr, password=pwd, domain=domain, my_name=local_name, remote_name=self.server_name, is_direct_tcp=True)
    def connect_to_server(self,timeout=60):
        '''
        This method establishes the connection to the NAS.
        
        Parameters 
        
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        is_connected = self.conn.connect(self.server_name,timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        return self.conn
    
    def read_files(self, remote_folder_path, timeout=60):
        '''
        This method reads all files from a NAS directory
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(self.server_name,timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Iterate in the folder to download all tif files
        list_files =[]
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            list_files.append(file.filename)
        return list_files

    def download_file(self, remote_file_path, local_folder_path, timeout=600):
        '''
        This method download an specific file
        
        Parameters
        
        remote_file_path : str, Pathlib obj
            The path in the remote file to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(self.server_name,timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_file_path)==str:
            remote_file_path = pathlib.Path(remote_file_path)
        # Making the local directory
        if not (os.path.exists(local_folder_path)) :
            os.makedirs(str(local_folder_path))
        filename = remote_file_path.name
        fileobj = open(remote_file_path.name,'wb')
        self.conn.retrieveFile(self.share_name, str(remote_file_path), fileobj)
        fileobj.close()
        # moving files in the local computer
        shutil.move(pathlib.Path().absolute().joinpath(filename), local_folder_path.joinpath(filename))
        print('Files downloaded to: ' + str(local_folder_path.joinpath(filename)))
        return None
    
    def copy_files(self, remote_folder_path, local_folder_path, timeout=600, file_extension ='.tif'):
        '''
        This method downloads tif files from NAS to a temporal folder in the local computer.
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        file_extension : str, optional.
            String representing the file type to download.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(self.server_name,timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Making the local directory
        if (os.path.exists(local_folder_path))  and  (str(local_folder_path.name)[0:5] ==  'temp_'):
            shutil.rmtree(local_folder_path)
        os.makedirs(str(local_folder_path))
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            if (file.filename not in ['.', '..']) and (file_extension in file.filename)  :
                print ('File Downloaded :', file.filename)
                fileobj = open(file.filename,'wb')
                self.conn.retrieveFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(file.filename) ),fileobj)
                fileobj.close()
                # moving files in the local computer
                shutil.move(pathlib.Path().absolute().joinpath(file.filename), local_folder_path.joinpath(file.filename))
        print('Files downloaded to: ' + str(local_folder_path))
        return None
    
    def write_files_to_NAS(self, local_file_to_send_to_NAS, remote_folder_path,  timeout=600):
        '''
        This method writes files from a local computer to NAS 
        
        Parameters
        
        local_file_to_send_to_NAS : str, Pathlib obj
            The path in the file to send to the NAS.
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(self.server_name,timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_file_to_send_to_NAS) == str:
            local_file_to_send_to_NAS = pathlib.Path(local_file_to_send_to_NAS)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # checks that the file doesn't exist on NAS. If it exist it will create a new name as follows original_name__1
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        list_all_files_in_NAS = [file.filename for file in list_dir]
        if str(local_file_to_send_to_NAS.name) not in list_all_files_in_NAS:
            with open(str(local_file_to_send_to_NAS), 'rb') as file_obj:
                self.conn.storeFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name) ) ,  file_obj )
                print ('The file was uploaded to NAS in location:', str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name))  )
        return None


class ReadImages():
    '''
    This class reads all tif images in a given folder and returns each image as a Numpy array inside a list, the names of these files, path, and the number of files.
    
    Parameters
    
    directory: str or PosixPath
        Directory containing the images to read.
    '''    
    def __init__(self, directory:str ):
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
            self.directory = pathlib.Path(directory)
    def read(self):
        '''
        Method takes all the images in the folder and merges those with similar names.
        
        Returns
        
        list_images : List of NumPy arrays 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C] or [T, Y, X, C]. 
        path_files : List of strings 
            List of strings containing the path to each image.
        list_files_names : List of strings 
            List of strings where each element is the name of the files in the directory.
        number_files : int 
            The number of images in the folder.
        '''
        list_files_names = sorted([f for f in listdir(self.directory) if isfile(join(self.directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files = [ str(self.directory.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
        number_files = len(path_files)
        list_images = [imread(str(f)) for f in path_files]
        return list_images, path_files, list_files_names, number_files


class MergeChannels():
    '''
    This class takes images as arrays with format [Z, Y, X] and merges them in a NumPy array with format [Z, Y, X, C].
    It recursively merges the channels in a new dimension in the array. The minimum number of channels 2 maximum is 4.
    
    Parameters

    directory: str or PosixPath
        Directory containing the images to merge.
    substring_to_detect_in_file_name: str
        String with the prefix to detect the names of the files. 
    save_figure: bool, optional
        If True, it saves the merged images as tif. The default is False. 
    '''
    def __init__(self, directory:str ,substring_to_detect_in_file_name:str = '.*_C0.tif', save_figure:bool = False ):
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
            self.directory = pathlib.Path(directory)
        self.substring_to_detect_in_file_name = substring_to_detect_in_file_name
        self.save_figure=save_figure

    def checking_images(self):
        '''
        Method that reads all images in the folder and returns a flag indicating if each channel in the image is separated in an independent file.
        
        Returns
        
        Flag : Bool 
            If True, it indicates that each channel is split into different files. If False, it indicates that the image is contained in a single file.
        '''
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file) and file[0]!= '.' : # detecting a match in the end, not consider hidden files starting with '.'
                    prefix = file.rpartition('_')[0]            # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif')) # List of files that match the pattern 'file_prefix_C*.tif'
                    if len(list_files_per_image)>1:             # creating merged files if more than one images with the same ending substring are detected.
                        return True
                    else:
                        return False
    
    def merge(self):
        '''
        Method takes all the images in the folder and merges those with similar names.
        
        Returns
        
        list_file_names : List of strings 
            List with strings of names.
        list_merged_images : List of NumPy arrays
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C].
        number_files : int
            The number of merged images in the folder.
        '''
        list_file_names =[]
        list_merged_images =[]  # list that stores all files belonging to the same image in a sublist
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        save_to_path = self.directory.joinpath('merged')
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file) and file[0]!= '.': # detecting a match in the end, not consider hidden files starting with '.'
                    prefix = file.rpartition('_')[0]            # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif') ) # List of files that match the pattern 'file_prefix_C*.tif'
                    if len(list_files_per_image)>1:             # creating merged files if more than one image with the exact ending substring is detected.
                        try:
                            list_file_names.append(prefix)
                            merged_img = np.concatenate([ imread(list_files_per_image[i])[..., np.newaxis] for i,_ in enumerate(list_files_per_image)],axis=-1).astype('uint16')
                            list_merged_images.append(merged_img) 
                        except:
                            print('A corrupted file was detected during the merge of the following files: \n', list_files_per_image,'\n')
                    if self.save_figure ==1 and len(list_files_per_image)>1:
                        if not os.path.exists(str(save_to_path)):
                            os.makedirs(str(save_to_path))
                        tifffile.imsave(str(save_to_path.joinpath(prefix+'_merged'+'.tif')), merged_img, metadata={'axes': 'ZYXC'})
        number_files = len(list_file_names)
        return list_file_names, list_merged_images, number_files,save_to_path


class RemoveExtrema():
    '''
    This class is intended to remove extreme values from an image. The format of the image must be [Y, X], [Y, X, C], or [Z, Y, X, C].
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Y, X], [Y, X, C], or [Z, Y, X, C].
    min_percentile : float, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : float, optional
        Higher bound to normalize intensity. The default is 99.
    selected_channels : List or None, optional
        Use this option to select a list channels to remove extrema. The default is None and applies the removal of extrema to all the channels.
    '''
    def __init__(self, image:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels = None):
        self.image = image
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        if not (type(selected_channels) is list):
            self.selected_channels = [selected_channels]
        else:
            self.selected_channels =selected_channels

    def remove_outliers(self):
        '''
        This method normalizes the values of an image by removing extreme values.
        
        Returns
        
        normalized_image : np.uint16
            Normalized image. Array with dimensions [Y, X, C], [Y, X], or [Z, Y, X, C].
        '''
        normalized_image = np.copy(self.image)
        normalized_image = np.array(normalized_image, 'float32')
        # Normalization code for image with format [Y, X]
        if len(self.image.shape) == 2:
            number_timepoints = 1
            number_channels = 1
            normalized_image_temp = normalized_image
            if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                max_val = np.percentile(normalized_image_temp, self.max_percentile)
                min_val = np.percentile(normalized_image_temp, self.min_percentile)
                normalized_image_temp [normalized_image_temp > max_val] = max_val
                normalized_image_temp [normalized_image_temp < min_val] = min_val
                normalized_image_temp [normalized_image_temp < 0] = 0
        # Normalization for image with format [Y, X, C].
        if len(self.image.shape) == 3:
            number_channels   = self.image.shape[2]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    normalized_image_temp = normalized_image[ :, :, index_channels]
                    if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                        max_val = np.percentile(normalized_image_temp, self.max_percentile)
                        min_val = np.percentile(normalized_image_temp, self.min_percentile)
                        normalized_image_temp [normalized_image_temp > max_val] = max_val
                        normalized_image_temp [normalized_image_temp < min_val] =  min_val
                        normalized_image_temp [normalized_image_temp < 0] = 0
        # Normalization for image with format [Z, Y, X, C].
        if len(self.image.shape) == 4:
            number_timepoints, number_channels   = self.image.shape[0], self.image.shape[3]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    for index_time in range (number_timepoints):
                        normalized_image_temp = normalized_image[index_time, :, :, index_channels]
                        if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                            max_val = np.percentile(normalized_image_temp, self.max_percentile)
                            min_val = np.percentile(normalized_image_temp, self.min_percentile)
                            normalized_image_temp [normalized_image_temp > max_val] = max_val
                            normalized_image_temp [normalized_image_temp < min_val] = min_val
                            normalized_image_temp [normalized_image_temp < 0] = 0
        return np.array(normalized_image, 'uint16')


class Cellpose():
    '''
    This class is intended to detect cells by image masking using `Cellpose <https://github.com/MouseLand/cellpose>`_ . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.
    For a complete description of Cellpose check the `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [T, Y, X, C].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0, 0], for RGB images with intensity for cytosol and nuclei use [0, 1] . The default is [0, 0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        Cellpose model type the options are 'cyto' for cytosol or 'nuclei' for the nucleus. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells' or 'max_cells_and_area'. The default is 'max_cells_and_area'.
    NUMBER_OF_CORES : int, optional
        The number of CPU cores to use for parallel computing. The default is 1.
    use_brute_force : bool, optional
        Flag to perform cell segmentation using all possible combination of parameters. The default is False.
    '''
    def __init__(self, image:np.ndarray, num_iterations:int = 4, channels:list = [0, 0], diameter:float = 120, model_type:str = 'cyto', selection_method:str = 'max_cells_and_area', NUMBER_OF_CORES:int=1, use_brute_force=False):
        self.image = image
        self.num_iterations = num_iterations
        #self.minimum_probability = -4
        #self.maximum_probability = 4
        self.minimum_probability = 0.1
        self.maximum_probability = 0.9
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        #self.CELLPOSE_PROBABILITY =  0.6
        #self.CELLPOSE_PROBABILITY =  0
        self.default_flow_threshold = 0.4 # default is 0.4
        self.optimization_parameter = np.unique(  np.round(np.linspace(self.minimum_probability, self.maximum_probability, self.num_iterations), 1) )
        self.use_brute_force = use_brute_force
        self.MINIMUM_CELL_AREA = 2000
    def calculate_masks(self):
        '''
        This method performs the process of image masking using **Cellpose**.
        
        Returns
        
        selected_masks : List of NumPy arrays
            List of NumPy arrays with values between 0 and the number of detected cells in the image, where an integer larger than zero represents the masked area for each cell, and 0 represents the background in the image.
        '''
        # Next two lines suppressing output from cellpose
        gc.collect()
        torch.cuda.empty_cache() 
        #old_stdout = sys.stdout
        #sys.stdout = open(os.devnull, "w")
        number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
        if number_gpus ==0:
            model = models.Cellpose(gpu = 0, model_type = self.model_type, omni = True) # model_type = 'cyto' or model_type = 'nuclei'
        else:
            model = models.Cellpose(gpu = 1, model_type = self.model_type, omni = True) # model_type = 'cyto' or model_type = 'nuclei'
        # Loop that test multiple probabilities in cell pose and returns the masks with the longest area.
        def cellpose_max_area( optimization_parameter):
            try:
                masks= model.eval(self.image, normalize = True, flow_threshold = optimization_parameter, diameter = self.diameter, min_size = self.MINIMUM_CELL_AREA, channels = self.channels, progress = None)[0]
            except:
                masks = 0
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(masks == nm)) # creating a list with the size of each mask
                largest_mask = np.argmax(size_mask)+1 # detecting the mask with the largest value
                temp_mask = np.zeros_like(masks) # making a copy of the image
                selected_mask = temp_mask + (masks == largest_mask) # Selecting a single mask and making this mask equal to one and the background equal to zero.
                return np.sum(selected_mask)
            else: # do nothing if only a single mask is detected per image.
                return np.sum(masks)
        def cellpose_max_cells(optimization_parameter):
            try:
                masks = model.eval(self.image, normalize = True, flow_threshold = optimization_parameter, diameter =self.diameter, min_size = self.MINIMUM_CELL_AREA, channels = self.channels, progress = None)[0]
            except:
                masks =0
            return np.max(masks)
        
        def cellpose_max_cells_and_area( optimization_parameter):
            try:
                masks = model.eval(self.image, normalize = True, flow_threshold = optimization_parameter,diameter = self.diameter, min_size = self.MINIMUM_CELL_AREA, channels = self.channels, progress = None)[0]
            except:
                masks = 0
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    approximated_radius = np.sqrt(np.sum(masks == nm)/np.pi)  # a=  pi r2
                    size_mask.append(approximated_radius) # creating a list with the size of each mask
                number_masks= np.max(masks)
                size_masks_array = np.array(size_mask)
                metric = np.median(size_masks_array).astype(int) * number_masks
                return metric
            if n_masks == 1: # do nothing if only a single mask is detected per image.
                return np.sum(masks == 1)
            else:  # return zero if no mask are detected
                return 0     
        if self.selection_method == 'max_area':
            list_metrics_masks = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(cellpose_max_area)(tested_parameter) for _, tested_parameter in enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        if self.selection_method == 'max_cells':
            list_metrics_masks = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(cellpose_max_cells)(tested_parameter) for _,tested_parameter in enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        if self.selection_method == 'max_cells_and_area':
            list_metrics_masks = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(cellpose_max_cells_and_area)(tested_parameter) for _,tested_parameter in enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        if not (self.selection_method is None) and (np.max(evaluated_metric_for_masks) >0) :
            selected_conditions = self.optimization_parameter[np.argmax(evaluated_metric_for_masks)]
            selected_masks = model.eval(self.image, normalize = True, flow_threshold = selected_conditions , diameter = self.diameter, min_size = self.MINIMUM_CELL_AREA, channels = self.channels, progress = None)[0]
        else:
            if len(self.image.shape) >= 3:
                selected_masks = np.zeros_like(self.image[:,:,0])
            else:
                selected_masks = np.zeros_like(self.image[:,:])
            print('No cells detected on the image')
        # If no GPU is available, the segmentation is performed with a single threshold. 
        if self.selection_method == None:
            selected_masks= model.eval(self.image, normalize = True, flow_threshold = self.default_flow_threshold, diameter = self.diameter, min_size = self.MINIMUM_CELL_AREA, channels = self.channels, progress = None)[0]
        return selected_masks
    
    
class CellSegmentation():
    '''
    This class is intended to detect cells in FISH images using `Cellpose <https://github.com/MouseLand/cellpose>`_ . This class segments the nucleus and cytosol for every cell detected in the image. The class uses optimization to generate the meta-parameters used by cellpose. For a complete description of Cellpose check the `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] or maximum projection with dimensions [Y, X, C].    
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    diameter_cytosol : int, optional
        Average cytosol size in pixels. The default is 150.
    diameter_nucleus : int, optional
        Average nucleus size in pixels. The default is 100.
    optimization_segmentation_method: str
        Method used for the segmentation. The options are: \'intensity_segmentation\', \'z_slice_segmentation\', \'gaussian_filter_segmentation\', and None.
    remove_fragmented_cells: bool, optional
        If true, it removes masks in the border of the image. The default is False.
    show_plots : bool, optional
        If true, it shows a plot with the detected masks. The default is True.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    use_brute_force : bool, optional
        Flag to perform cell segmentation using all possible combination of parameters. The default is False.
    '''
    def __init__(self, image:np.ndarray, channels_with_cytosol = None, channels_with_nucleus= None, diameter_cytosol:float = 150, diameter_nucleus:float = 100, optimization_segmentation_method='z_slice_segmentation', remove_fragmented_cells:bool=False, show_plots: bool = True, image_name = None,use_brute_force=False):
        self.image = image
        self.channels_with_cytosol = channels_with_cytosol
        self.channels_with_nucleus = channels_with_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.diameter_nucleus = diameter_nucleus
        self.show_plots = show_plots
        self.remove_fragmented_cells = remove_fragmented_cells
        self.image_name = image_name
        self.NUMBER_OF_CORES = 1 # number of cores for parallel computing.
        self.use_brute_force = use_brute_force
        number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
        self.number_z_slices = image.shape[0]
        if number_gpus ==0:
            self.NUMBER_OPTIMIZATION_VALUES= 0
            self.optimization_segmentation_method = None # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
        else:
            self.NUMBER_OPTIMIZATION_VALUES= np.min((self.number_z_slices,15))
            self.optimization_segmentation_method = optimization_segmentation_method  # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
        if self.optimization_segmentation_method == 'z_slice_segmentation_marker':
            self.NUMBER_OPTIMIZATION_VALUES= self.number_z_slices
        
    def calculate_masks(self):
        '''
        This method performs the process of cell detection for FISH images using **Cellpose**.
        
        Returns
        
        masks_complete_cells : NumPy array. np.uint8
            Image containing the masks for every cell detected in the image. Numpy array with format [Y, X].
        masks_nuclei : NumPy array. np.uint8
            Image containing the masks for every nuclei detected in the image. Numpy array with format [Y, X].
        masks_cytosol_no_nuclei : NumPy array. np.uint8
            Image containing the masks for every cytosol (removing the nucleus) detected in the image. Numpy array with format [Y, X].
        '''
        # function that determines if the nucleus is in the cytosol
        def is_nucleus_in_cytosol(mask_n, mask_c):
            mask_n[mask_n>1]=1
            mask_c[mask_c>1]=1
            size_mask_n = np.count_nonzero(mask_n)
            size_mask_c = np.count_nonzero(mask_c)
            min_size =np.min( (size_mask_n,size_mask_c) )
            mask_combined =  mask_n + mask_c
            sum_mask = np.count_nonzero(mask_combined[mask_combined==2])
            if (sum_mask> min_size*0.8) and (min_size>2500): # the element is inside if the two masks overlap over the 80% of the smaller mask.
                return 1
            else:
                return 0
        ##### IMPLEMENTATION #####
        if len(self.image.shape) > 3:  # [ZYXC]
            if self.image.shape[0] ==1:
                image_normalized = self.image[0,:,:,:]
            else:
                center_slice = self.number_z_slices//2
                image_normalized = self.image[center_slice,:,:,:]
                #image_normalized = np.max(self.image[3:-3,:,:,:],axis=0)    # taking the mean value
        else:
            image_normalized = self.image # [YXC] 
        
        # Function to calculate the approximated radius in a mask
        def approximated_radius(masks,diameter=100):
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    approximated_radius = np.sqrt(np.sum(masks == nm)/np.pi)  # a=  pi r2
                    #if  approximated_radius < diameter:
                    size_mask.append(approximated_radius) # creating a list with the size of each mask
                    #else:
                    #    size_mask.append(0)
                array_radii =np.array(size_mask)
            else:
                array_radii = np.sqrt(np.sum(masks == 1)/np.pi)
            if np.any(array_radii > diameter*1.5) |  np.any(array_radii < 10):
                masks_radii = 0
            else:
                masks_radii= np.prod (array_radii)
            return masks_radii
        
        # Function to find masks
        def function_to_find_masks (image):                    
            if not (self.channels_with_cytosol is None):
                masks_cyto = Cellpose(image[:, :, self.channels_with_cytosol],diameter = self.diameter_cytosol, model_type = 'cyto', selection_method = 'max_cells_and_area' ,NUMBER_OF_CORES=self.NUMBER_OF_CORES,use_brute_force=self.use_brute_force).calculate_masks()
            else:
                masks_cyto = np.zeros_like(image[:, :, 0])
            if not (self.channels_with_nucleus is None):
                masks_nuclei = Cellpose(image[:, :, self.channels_with_nucleus],  diameter = self.diameter_nucleus, model_type = 'nuclei', selection_method = 'max_cells_and_area',NUMBER_OF_CORES=self.NUMBER_OF_CORES,use_brute_force=self.use_brute_force).calculate_masks()
            else:
                masks_nuclei= np.zeros_like(image[:, :, 0])
            if not (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                # Function that removes masks that are not paired with a nucleus or cytosol
                def remove_lonely_masks(masks_0, masks_1,is_nuc=None):
                    n_mask_0 = np.max(masks_0)
                    n_mask_1 = np.max(masks_1)
                    if (n_mask_0>0) and (n_mask_1>0):
                        for ind_0 in range(1,n_mask_0+1):
                            tested_mask_0 = np.where(masks_0 == ind_0, 1, 0)
                            array_paired= np.zeros(n_mask_1)
                            for ind_1 in range(1,n_mask_1+1):
                                tested_mask_1 = np.where(masks_1 == ind_1, 1, 0)
                                array_paired[ind_1-1] = is_nucleus_in_cytosol(tested_mask_1, tested_mask_0)
                                #remove_lonely_masks(masks_nuclei, masks_cyto,is_nuc='nuc')
                                if (is_nuc =='nuc') and (np.count_nonzero(tested_mask_0) > np.count_nonzero(tested_mask_1) ):
                                    # condition that rejects images with nucleus bigger than the cytosol
                                    array_paired[ind_1-1] = 0
                                elif (is_nuc is None ) and (np.count_nonzero(tested_mask_1) > np.count_nonzero(tested_mask_0) ):
                                    array_paired[ind_1-1] = 0
                            if any (array_paired) == False: # If the cytosol is not associated with any mask.
                                masks_0 = np.where(masks_0 == ind_0, 0, masks_0)
                            masks_with_pairs = masks_0
                    else:
                        masks_with_pairs = np.zeros_like(masks_0)
                    return masks_with_pairs
                # Function that reorder the index to make it continuos 
                def reorder_masks(mask_tested):
                    n_mask_0 = np.max(mask_tested)
                    mask_new =np.zeros_like(mask_tested)
                    if n_mask_0>0:
                        counter = 0
                        for ind_0 in range(1,n_mask_0+1):
                            if ind_0 in mask_tested:
                                counter = counter + 1
                                if counter ==1:
                                    mask_new = np.where(mask_tested == ind_0, -counter, mask_tested)
                                else:
                                    mask_new = np.where(mask_new == ind_0, -counter, mask_new)
                        reordered_mask = np.absolute(mask_new)
                    else:
                        reordered_mask = mask_new
                    return reordered_mask                
                # Cytosol masks
                masks_cyto = remove_lonely_masks(masks_cyto, masks_nuclei)
                masks_cyto = reorder_masks(masks_cyto)
                # Masks nucleus
                masks_nuclei = remove_lonely_masks(masks_nuclei, masks_cyto,is_nuc='nuc')
                masks_nuclei = reorder_masks(masks_nuclei)
                # Iterate for each cyto mask
                def matching_masks(masks_cyto,masks_nuclei):
                    n_mask_cyto = np.max(masks_cyto)
                    n_mask_nuc = np.max(masks_nuclei)
                    new_masks_nuclei = np.zeros_like(masks_cyto)
                    reordered_mask_nuclei = np.zeros_like(masks_cyto)
                    for mc in range(1,n_mask_cyto+1):
                        tested_mask_cyto = np.where(masks_cyto == mc, 1, 0)
                        for mn in range(1,n_mask_nuc+1):
                            mask_paired = False
                            tested_mask_nuc = np.where(masks_nuclei == mn, 1, 0)
                            mask_paired = is_nucleus_in_cytosol(tested_mask_nuc, tested_mask_cyto)
                            if mask_paired == True:
                                if np.count_nonzero(new_masks_nuclei) ==0:
                                    new_masks_nuclei = np.where(masks_nuclei == mn, -mc, masks_nuclei)
                                else:
                                    new_masks_nuclei = np.where(new_masks_nuclei == mn, -mc, new_masks_nuclei)
                        reordered_mask_nuclei = np.absolute(new_masks_nuclei)
                    return masks_cyto,reordered_mask_nuclei
                # Matching nuclei and cytosol
                masks_cyto, masks_nuclei = matching_masks(masks_cyto,masks_nuclei)                
                #Generating mask for cyto without nuc
                masks_cytosol_no_nuclei = masks_cyto-masks_nuclei
                masks_cytosol_no_nuclei[masks_cytosol_no_nuclei < 0] = 0
                masks_cytosol_no_nuclei.astype(int)
                # Renaming 
                masks_complete_cells = masks_cyto
            else:
                if not (self.channels_with_cytosol is None):
                    masks_complete_cells = masks_cyto
                    masks_nuclei = None 
                    masks_cytosol_no_nuclei = None
                if not (self.channels_with_nucleus is None):
                    masks_complete_cells = masks_nuclei
                    masks_cytosol_no_nuclei = None
            return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei
        # OPTIMIZATION METHODS FOR SEGMENTATION
        if (self.optimization_segmentation_method == 'intensity_segmentation') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Intensity Based Optimization to find the maximum number of index_paired_masks. 
            if not (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                tested_thresholds = np.round(np.linspace(0, 3, self.NUMBER_OPTIMIZATION_VALUES), 0)
                list_sorting_number_paired_masks = []
                for _, threshold in enumerate(tested_thresholds):
                    image_copy = image_normalized.copy()
                    image_temp = RemoveExtrema(image_copy,min_percentile=threshold, max_percentile=100-threshold,selected_channels=self.channels_with_cytosol).remove_outliers() 
                    masks_complete_cells, _, _ = function_to_find_masks(image_temp)
                    list_sorting_number_paired_masks.append(np.max(masks_complete_cells))
                array_number_paired_masks = np.array(list_sorting_number_paired_masks)
                selected_threshold = tested_thresholds[np.argmax(array_number_paired_masks)]
            else:
                selected_threshold = 0
            # Running the mask selection once a threshold is obtained
            image_copy = image_normalized.copy()
            image_temp = RemoveExtrema(image_copy,min_percentile=selected_threshold,max_percentile=100-selected_threshold,selected_channels=self.channels_with_cytosol).remove_outliers() 
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(image_temp)        
        
        elif (self.optimization_segmentation_method == 'z_slice_segmentation_marker') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks. 
            list_idx = np.round(np.linspace(0, self.number_z_slices-1, self.NUMBER_OPTIMIZATION_VALUES), 0).astype(int)  
            list_idx = np.unique(list_idx)  #list(set(list_idx))
            # Optimization based on slice
            if not (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                list_sorting_number_paired_masks = []
                array_number_paired_masks = np.zeros( len(list_idx) )
                # performing the segmentation on a maximum projection
                test_image_optimization = np.max(self.image[:,:,:,:],axis=0)  
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)                
                median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,diameter=self.diameter_cytosol)
                median_radii_nuclei = approximated_radius(masks_nuclei,diameter=self.diameter_nucleus)
                metric_max =  median_radii_nuclei * median_radii_complete_cells_complete_cells
                # performing segmentation for a subsection of z-slices
                for idx, idx_value in enumerate(list_idx):
                    test_image_optimization = self.image[idx_value,:,:,:]
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,diameter=self.diameter_cytosol)
                    median_radii_nuclei = approximated_radius(masks_nuclei,diameter=self.diameter_nucleus)
                    metric = median_radii_nuclei * median_radii_complete_cells_complete_cells
                    array_number_paired_masks[idx] = metric
                if np.any (array_number_paired_masks> metric_max):
                    selected_index = list_idx[np.argmax(array_number_paired_masks)]
                else:
                    selected_index = None
            else:
                selected_index = None
            # Running the mask selection once a threshold is obtained
            if not(selected_index is None):
                test_image_optimization = self.image[selected_index,:,:,:]
            else:
                test_image_optimization = np.max(self.image[:,:,:,:],axis=0)  
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization) 
        
        
        elif (self.optimization_segmentation_method == 'z_slice_segmentation') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks. 
            num_slices_range = 3  # range to consider above and below a selected z-slice
            list_idx = np.round(np.linspace(num_slices_range, self.number_z_slices-num_slices_range, self.NUMBER_OPTIMIZATION_VALUES), 0).astype(int)  
            list_idx = np.unique(list_idx)  #list(set(list_idx))
            # Optimization based on slice
            if not (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                list_sorting_number_paired_masks = []
                array_number_paired_masks = np.zeros( len(list_idx) )
                # performing the segmentation on a maximum projection
                test_image_optimization = np.max(self.image[num_slices_range:-num_slices_range,:,:,:],axis=0)  
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)                
                median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,diameter=self.diameter_cytosol)
                median_radii_nuclei = approximated_radius(masks_nuclei,diameter=self.diameter_nucleus)
                metric_max =  median_radii_nuclei * median_radii_complete_cells_complete_cells
                # performing segmentation for a subsection of z-slices
                for idx, idx_value in enumerate(list_idx):
                    test_image_optimization = np.max(self.image[idx_value-num_slices_range:idx_value+num_slices_range,:,:,:],axis=0)  
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,diameter=self.diameter_cytosol)
                    median_radii_nuclei = approximated_radius(masks_nuclei,diameter=self.diameter_nucleus)
                    metric =  median_radii_nuclei * median_radii_complete_cells_complete_cells
                    try:
                        array_number_paired_masks[idx] = metric
                    except:
                        array_number_paired_masks[idx] = 0
                if np.any (array_number_paired_masks> metric_max):
                    selected_index = list_idx[np.argmax(array_number_paired_masks)]
                else:
                    selected_index = None
            else:
                selected_index = None
            # Running the mask selection once a threshold is obtained
            if not(selected_index is None):
                test_image_optimization = np.max(self.image[selected_index-num_slices_range:selected_index+num_slices_range,:,:,:],axis=0) 
            else:
                test_image_optimization = np.max(self.image[num_slices_range:-num_slices_range,:,:,:],axis=0)  
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)        
        
                
        elif (self.optimization_segmentation_method == 'center_slice') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks. 
            #number_z_slices = self.image.shape[0]
            center_slice = self.number_z_slices//2
            num_slices_range = np.min( (5,center_slice-1))  # range to consider above and below a selected z-slice
            # Optimization based on slice
            #test_image_optimization = np.max(self.image[center_slice-num_slices_range:center_slice+num_slices_range,:,:,:],axis=0) 
            test_image_optimization = self.image[center_slice,:,:,:] 
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
        elif (self.optimization_segmentation_method == 'gaussian_filter_segmentation') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on testing different sigmas in a gaussian filter to find the maximum number of index_paired_masks. 
            half_z_slices = self.image.shape[0]//2
            list_sigmas = np.round(np.linspace(0.5, 20, self.NUMBER_OPTIMIZATION_VALUES), 1) 
            # Optimization based on slice
            if not (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                list_sorting_number_paired_masks = []
                array_number_paired_masks = np.zeros( len(list_sigmas)  )
                for idx, sigma_value in enumerate(list_sigmas):
                    test_image_optimization = stack.gaussian_filter(self.image[half_z_slices,:,:,:],sigma=sigma_value)  
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    metric = np.max(masks_complete_cells)
                    array_number_paired_masks[idx] = metric
                selected_threshold = list_sigmas[np.argmax(array_number_paired_masks)]
            else:
                selected_threshold = list_sigmas[0]
            # Running the mask selection once a threshold is obtained
            test_image_optimization = stack.gaussian_filter(self.image[half_z_slices,:,:,:],sigma=selected_threshold) 
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
        else:
            # no optimization is applied if a 2D image is passed
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(image_normalized)
        # This functions makes zeros the border of the mask, it is used only for plotting.
        def remove_border(img,px_to_remove = 1):
            img[0:px_to_remove, :] = 0;img[:, 0:px_to_remove] = 0;img[img.shape[0]-px_to_remove:img.shape[0]-1, :] = 0; img[:, img.shape[1]-px_to_remove: img.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
            return img
        # Plotting
        if np.max(masks_complete_cells) != 0 and not(self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
            
            n_channels = np.min([3, image_normalized.shape[2]])
            _, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 10))
            im = Utilities().convert_to_int8(image_normalized[ :, :, 0:n_channels],min_percentile=1, max_percentile=95)  
            masks_plot_cyto= masks_complete_cells 
            masks_plot_nuc = masks_nuclei              
            axes[0].imshow(im)
            axes[0].set(title = 'All channels')
            axes[1].imshow(masks_plot_cyto)
            axes[1].set(title = 'Cytosol mask')
            axes[2].imshow(masks_plot_nuc)
            axes[2].set(title = 'Nuclei mask')
            axes[3].imshow(im)
            n_masks =np.max(masks_complete_cells)                 
            for i in range(1, n_masks+1 ):
                # Removing the borders just for plotting
                tested_mask_cyto = np.where(masks_complete_cells == i, 1, 0).astype(bool)
                tested_mask_nuc = np.where(masks_nuclei == i, 1, 0).astype(bool)
                # Remove border for plotting
                temp_nucleus_mask= remove_border(tested_mask_nuc)
                temp_complete_mask = remove_border(tested_mask_cyto)
                temp_contour_n = find_contours(temp_nucleus_mask, 0.1, fully_connected='high')
                temp_contour_c = find_contours(temp_complete_mask, 0.1, fully_connected='high')
                contours_connected_n = np.vstack((temp_contour_n))
                contour_n = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                contours_connected_c = np.vstack((temp_contour_c))
                contour_c = np.vstack((contours_connected_c[-1,:],contours_connected_c))
                axes[3].fill(contour_n[:, 1], contour_n[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask nucleus
                axes[3].fill(contour_c[:, 1], contour_c[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask cytosol
                axes[3].set(title = 'Paired masks')
            if not(self.image_name is None):
                plt.savefig(self.image_name,bbox_inches='tight')
            if self.show_plots == 1:
                plt.show()
            else:
                plt.close()
        else:
            if not(self.channels_with_cytosol is None) and (self.channels_with_nucleus is None):
                masks_plot_cyto= masks_complete_cells 
                n_channels = np.min([3, image_normalized.shape[2]])
                _, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
                im = Utilities().convert_to_int8(image_normalized[ :, :, 0:n_channels],min_percentile=1, max_percentile=95)
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_cyto)
                axes[1].set(title = 'Cytosol mask')
                if not(self.image_name is None):
                    plt.savefig(self.image_name,bbox_inches='tight')
                if self.show_plots == 1:
                    plt.show()
                else:
                    plt.close()
            if (self.channels_with_cytosol is None) and not(self.channels_with_nucleus is None):
                masks_plot_nuc = masks_nuclei    
                n_channels = np.min([3, image_normalized.shape[2]])
                _, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
                im = Utilities().convert_to_int8(image_normalized[ :, :, 0:n_channels],min_percentile=1, max_percentile=95)
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_nuc)
                axes[1].set(title = 'Nuclei mask')
                if not(self.image_name is None):
                    plt.savefig(self.image_name,bbox_inches='tight')
                if self.show_plots == 1:
                    plt.show()
                else:
                    plt.close()
            print('No paired masks were detected for this image')
        
        return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei 


class BigFISH():
    '''
    This class is intended to detect spots in FISH images using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. The format of the image must be  [Z, Y, X, C].
    
    Parameters
    
    The description of the parameters is taken from `Big-FISH <https://github.com/fish-quant/big-fish>`_ BSD 3-Clause License. Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    FISH_channel : int
        Specific channel with FISH spots that are used for the quantification
    voxel_size_z : int, optional
        Height of a voxel, along the z axis, in nanometers. The default is 300.
    voxel_size_yx : int, optional
        Size of a voxel on the yx plan in nanometers. The default is 150.
    psf_z : int, optional
        Theoretical size of the PSF emitted by a spot in the z plan, in nanometers. The default is 350.
    psf_yx : int, optional
        Theoretical size of the PSF emitted by a spot in the yx plan in nanometers.
    cluster_radius : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    show_plots : bool, optional
        If True shows a 2D maximum projection of the image and the detected spots. The default is False
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is False.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calculated automatically.
    '''
    def __init__(self,image, FISH_channel , voxel_size_z = 300,voxel_size_yx = 103,psf_z = 350, psf_yx = 150, cluster_radius = 350,minimum_spots_cluster = 4,  show_plots =False,image_name=None,save_all_images=True,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None):
        self.image = image
        self.FISH_channel = FISH_channel
        self.voxel_size_z = voxel_size_z
        self.voxel_size_yx = voxel_size_yx
        self.psf_z = psf_z
        self.psf_yx = psf_yx
        self.cluster_radius = cluster_radius
        self.minimum_spots_cluster = minimum_spots_cluster
        self.show_plots = show_plots
        self.image_name=image_name
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        self.threshold_for_spot_detection=threshold_for_spot_detection
    def detect(self):
        '''
        This method is intended to detect RNA spots in the cell and Transcription Sites (Clusters) using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
        
        Returns
        
        clusterDetectionCSV : np.int64 Array with shape (nb_clusters, 5) or (nb_clusters, 4). 
            One coordinate per dimension for the centroid of the cluster (zyx or yx coordinates), the number of spots detected in the clusters, and its index.
        spotDetectionCSV :  np.int64 with shape (nb_spots, 4) or (nb_spots, 3).
            Coordinates of the detected spots. One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, the value is -1.
        '''
        rna=self.image[:,:,:,self.FISH_channel]
        # Calculating Sigma with  the parameters for the PSF.
        spot_radius_px = detection.get_object_radius_pixel(
                        voxel_size_nm=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx), 
                        object_radius_nm=(self.psf_z, self.psf_yx, self.psf_yx), ndim=3)
        sigma = spot_radius_px
        print('sigma_value (z,y,x) =', sigma)
        ## SPOT DETECTION
        if self.use_log_filter_for_spot_detection== True:
            try:
                rna_filtered = stack.log_filter(rna, sigma) # LoG filter
            except ValueError:
                print('Error during the log filter calculation, try using larger parameters values for the psf')
                rna_filtered = stack.remove_background_gaussian(rna, sigma)
        else:
            rna_filtered = stack.remove_background_gaussian(rna, sigma)
        # Automatic threshold detection.
        mask = detection.local_maximum_detection(rna_filtered, min_distance=sigma) # local maximum detection        
        if not (self.threshold_for_spot_detection is None):
            threshold = self.threshold_for_spot_detection
        else:
            threshold = detection.automated_threshold_setting(rna_filtered, mask) # thresholding
        print('Int threshold used for the detection of spots: ',threshold )
        spots, _ = detection.spots_thresholding(rna_filtered, mask, threshold, remove_duplicate=True)
        # Decomposing dense regions
        spots_post_decomposition, _, _ = detection.decompose_dense(image=rna, spots=spots, 
                                                                voxel_size = (self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx), 
                                                                spot_radius = (self.psf_z, self.psf_yx, self.psf_yx),
                                                                alpha=0.9,   # alpha impacts the number of spots per candidate region
                                                                beta=1,      # beta impacts the number of candidate regions to decompose
                                                                gamma=5)     # gamma the filtering step to denoise the image
        ### CLUSTER DETECTION
        spots_post_clustering, clusters = detection.detect_clusters(spots_post_decomposition, 
                                            voxel_size=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx),
                                            radius= self.cluster_radius, nb_min_spots = self.minimum_spots_cluster)
        # Saving results with new variable names
        spotDetectionCSV = spots_post_clustering
        clusterDetectionCSV = clusters
        ## PLOTTING
        #if self.show_plots == True:
        try:
            if not(self.image_name is None):
                path_output_elbow= str(self.image_name) +'__elbow_'+ '_ch_' + str(self.FISH_channel) + '.png'
            else:
                path_output_elbow = None 
            plot.plot_elbow(rna, 
                            voxel_size=(self.voxel_size_z, self.voxel_size_yx,self.voxel_size_yx), 
                            spot_radius= (self.psf_z, self.psf_yx, self.psf_yx),
                            path_output = path_output_elbow, show=self.show_plots )
            if self.show_plots ==True:
                plt.show()
            else:
                plt.close()
        except:
            print('not showing elbow plot')
        central_slice = rna.shape[0]//2
        if self.save_all_images:
            range_plot_images = range(0, rna.shape[0])
        else:
            range_plot_images = range(central_slice,central_slice+1)      
        for i in range_plot_images:
            if i==central_slice:
                print('Z-Slice: ', str(i))
            image_2D = rna_filtered[i,:,:]
            if i > 1 and i<rna.shape[0]-1:
                if self.display_spots_on_multiple_z_planes == True:
                    # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane-1
                    clusters_to_plot = clusters[(clusters[:,0]>=i-1) & (clusters[:,0]<=i+2) ] 
                    spots_to_plot =  spots_post_decomposition [(spots_post_decomposition[:,0]>=i-1) & (spots_post_decomposition[:,0]<=i+1) ]
                else:
                    clusters_to_plot = clusters[clusters[:,0]==i]
                    spots_to_plot =  spots_post_decomposition [spots_post_decomposition[:,0]==i ]
            else:
                clusters_to_plot = clusters[clusters[:,0]==i]
                spots_to_plot =  spots_post_decomposition [spots_post_decomposition[:,0]==i ]
            if self.save_all_images:
                path_output= str(self.image_name) + '_ch_' + str(self.FISH_channel) +'_slice_'+ str(i) +'.png'
            else:
                path_output= str(self.image_name) + '_ch_' + str(self.FISH_channel) +'.png'
                
            if not(self.image_name is None) and (i==central_slice) and (self.show_plots ==True): # saving only the central slice
                show_figure_in_cli = True
            else:
                show_figure_in_cli = False                
            plot.plot_detection(image_2D, 
                            spots=[spots_to_plot, clusters_to_plot[:, :3]], 
                            shape=["circle", "polygon"], 
                            radius=[3, 6], 
                            color=["orangered", "blue"],
                            linewidth=[1, 1], 
                            fill=[False, False], 
                            framesize=(15, 10), 
                            contrast=True,
                            rescale=True,
                            show=show_figure_in_cli,
                            path_output = path_output)
            if self.show_plots ==True:
                plt.show()
            else:
                plt.close()
            del spots_to_plot, clusters_to_plot
        return [spotDetectionCSV, clusterDetectionCSV]


class DataProcessing():
    '''
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter descriptions obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    spotDetectionCSV: np.int64 Array with shape (nb_clusters, 5) or (nb_clusters, 4). 
        One coordinate per dimension for the cluster\'s centroid (zyx or yx coordinates), the number of spots detected in the clusters, and its index.
    clusterDetectionCSV : np.int64 with shape (nb_spots, 4) or (nb_spots, 3).
        Coordinates of the detected spots . One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, the value is -1.
    masks_complete_cells : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_nuclei: List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    spot_type : int, optional
        A label indicating the spot type, this counter starts at zero, increasing with the number of channels containing FISH spots. The default is zero.
    dataframe : Pandas dataframe or None.
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nucleus_y, nucleus_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    reset_cell_counter : bool
        This number is used to reset the counter of the number of cells. The default is False.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    '''
    def __init__(self,spotDetectionCSV, clusterDetectionCSV,masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei,  channels_with_cytosol,channels_with_nucleus   , spot_type=0, dataframe =None,reset_cell_counter=False,image_counter=0):
        self.spotDetectionCSV=spotDetectionCSV 
        self.clusterDetectionCSV=clusterDetectionCSV
        self.channels_with_cytosol=channels_with_cytosol
        self.channels_with_nucleus=channels_with_nucleus
        if isinstance(masks_complete_cells, list) or (masks_complete_cells is None):
            self.masks_complete_cells=masks_complete_cells
        else:
            self.masks_complete_cells=Utilities().separate_masks(masks_complete_cells)
        if isinstance(masks_nuclei, list) or (masks_nuclei is None):
            self.masks_nuclei=masks_nuclei
        else:
            self.masks_nuclei=Utilities().separate_masks(masks_nuclei)        
        if isinstance(masks_cytosol_no_nuclei, list) or (masks_cytosol_no_nuclei is None):
            self.masks_cytosol_no_nuclei=masks_cytosol_no_nuclei
        else:
            self.masks_cytosol_no_nuclei= Utilities().separate_masks(masks_cytosol_no_nuclei)
        self.dataframe=dataframe
        self.spot_type = spot_type
        self.reset_cell_counter = reset_cell_counter
        self.image_counter = image_counter
    def get_dataframe(self):
        '''
        This method extracts data from the class SpotDetection and returns the data as a dataframe.
        
        Returns
        
        dataframe : Pandas dataframe
            Pandas dataframe with the following columns. image_id, cell_id, spot_id, nucleus_y, nucleus_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented.
        '''
        def mask_selector(mask,calculate_centroid= True):
            mask_area = np.count_nonzero(mask)
            if calculate_centroid == True:
                centroid_y,centroid_x = ndimage.measurements.center_of_mass(mask)
            else:
                centroid_y,centroid_x = 0,0
            return  mask_area, centroid_y, centroid_x
        def data_to_df( df, spotDetectionCSV, clusterDetectionCSV, mask_nuc = None, mask_cytosol_only=None, nuc_area = 0, cyto_area =0, cell_area=0, centroid_y=0, centroid_x=0, image_counter=0, is_cell_in_border = 0, spot_type=0, cell_counter =0 ):
            # spotDetectionCSV      nrna x  [Z,Y,X,idx_foci]
            # clusterDetectionCSV   nc   x  [Z,Y,X,size,idx_foci]
            # Removing TS from the image and calculating RNA in nucleus
            if not (self.channels_with_nucleus is None):
                spots_no_ts, _, ts = multistack.remove_transcription_site(spotDetectionCSV, clusterDetectionCSV, mask_nuc, ndim=3)
            else:
                spots_no_ts, ts = spotDetectionCSV, None
            #rna_out_ts      [Z,Y,X,idx_foci]         Coordinates of the detected RNAs with shape. One coordinate per dimension (zyx or yx coordinates) plus the index of the foci assigned to the RNA. If no foci was assigned, value is -1. RNAs from transcription sites are removed.
            #foci            [Z,Y,X,size, idx_foci]   One coordinate per dimension for the foci centroid (zyx or yx coordinates), the number of RNAs detected in the foci and its index.
            #ts              [Z,Y,X,size,idx_ts]      One coordinate per dimension for the transcription site centroid (zyx or yx coordinates), the number of RNAs detected in the transcription site and its index.
            if not (self.channels_with_nucleus is None):
                spots_nuc, _ = multistack.identify_objects_in_region(mask_nuc, spots_no_ts, ndim=3)
            else:
                spots_nuc = None
            # Detecting spots in the cytosol
            if not (self.channels_with_cytosol is None) :
                spots_cytosol_only, _ = multistack.identify_objects_in_region(mask_cytosol_only, spotDetectionCSV[:,:3], ndim=3)
            else:
                spots_cytosol_only = None
            # coord_innp  Coordinates of the objects detected inside the region.
            # coord_outnp Coordinates of the objects detected outside the region.
            # ts                     n x [Z,Y,X,size,idx_ts]
            # spots_nuc              n x [Z,Y,X]
            # spots_cytosol_only     n x [Z,Y,X]
            number_columns = len(df. columns)
            if not(spots_nuc is None):
                num_ts = ts.shape[0]
                num_nuc = spots_nuc.shape[0]
            else:
                num_ts = 0
                num_nuc = 0
            if not(spots_cytosol_only is None) :
                num_cyt = spots_cytosol_only.shape[0] 
            else:
                num_cyt = 0
            array_ts =                  np.zeros( ( num_ts,number_columns)  )
            array_spots_nuc =           np.zeros( ( num_nuc,number_columns) )
            array_spots_cytosol_only =  np.zeros( ( num_cyt  ,number_columns) )
            # Spot index 
            spot_idx_ts =  np.arange(0,                  num_ts                                                 ,1 )
            spot_idx_nuc = np.arange(num_ts,             num_ts + num_nuc                                       ,1 )
            spot_idx_cyt = np.arange(num_ts + num_nuc,   num_ts + num_nuc + num_cyt  ,1 )
            spot_idx = np.concatenate((spot_idx_ts,  spot_idx_nuc, spot_idx_cyt ))
            # Populating arrays
            if not (self.channels_with_nucleus is None):
                array_ts[:,8:11] = ts[:,:3]         # populating coord 
                array_ts[:,11] = 1                  # is_nuc
                array_ts[:,12] = 1                  # is_cluster
                array_ts[:,13] =  ts[:,3]           # cluster_size
                array_ts[:,14] = spot_type          # spot_type
                array_ts[:,15] = is_cell_in_border  # is_cell_fragmented
                array_spots_nuc[:,8:11] = spots_nuc[:,:3]   # populating coord 
                array_spots_nuc[:,11] = 1                   # is_nuc
                array_spots_nuc[:,12] = 0                   # is_cluster
                array_spots_nuc[:,13] = 0                   # cluster_size
                array_spots_nuc[:,14] =  spot_type          # spot_type
                array_spots_nuc[:,15] =  is_cell_in_border  # is_cell_fragmented
            if not (self.channels_with_cytosol is None) :
                array_spots_cytosol_only[:,8:11] = spots_cytosol_only[:,:3]    # populating coord 
                array_spots_cytosol_only[:,11] = 0                             # is_nuc
                array_spots_cytosol_only[:,12] = 0                             # is_cluster
                array_spots_cytosol_only[:,13] = 0                            # cluster_size
                array_spots_cytosol_only[:,14] =  spot_type                    # spot_type
                array_spots_cytosol_only[:,15] =  is_cell_in_border            # is_cell_fragmented
            # concatenate array
            if not (self.channels_with_nucleus is None) and not (self.channels_with_cytosol is None):
                array_complete = np.vstack((array_ts, array_spots_nuc, array_spots_cytosol_only))
            elif not (self.channels_with_nucleus is None) and (self.channels_with_cytosol is None):
                array_complete = np.vstack((array_ts, array_spots_nuc))
            elif (self.channels_with_nucleus is None) and not (self.channels_with_cytosol is None):
                array_complete = array_spots_cytosol_only
            # Saves a dataframe with zeros when no spots are detected on the cell.
            if array_complete.size ==0:
                array_complete = np.zeros( ( 1,number_columns)  )
                # if NO spots are detected populate  with -1
                array_complete[:,2] = -1     # spot_id
                array_complete[:,8:16] = -1
            else:
                # if spots are detected populate  the reported  array
                array_complete[:,2] = spot_idx.T     # spot_id
            # populating  array with cell  information
            array_complete[:,0] = image_counter  # image_id
            array_complete[:,1] = cell_counter   # cell_id
            array_complete[:,3] = centroid_y     #'nuc_y_centoid'
            array_complete[:,4] = centroid_x     #'nuc_x_centoid'
            array_complete[:,5] = nuc_area       #'nuc_area_px'
            array_complete[:,6] = cyto_area      # cyto_area_px
            array_complete[:,7] = cell_area      #'cell_area_px'
            df = df.append(pd.DataFrame(array_complete, columns=df.columns), ignore_index=True)
            new_dtypes = {'image_id':int, 'cell_id':int, 'spot_id':int,'is_nuc':int,'is_cluster':int,'nucleus_y':int, 'nucleus_x':int,'nuc_area_px':int,'cyto_area_px':int, 'cell_area_px':int,'x':int,'y':int,'z':int,'cluster_size':int,'spot_type':int,'is_cell_fragmented':int}
            df = df.astype(new_dtypes)
            return df
        if not (self.masks_nuclei is None):
            n_masks = len(self.masks_nuclei)
        else:
            n_masks = len(self.masks_complete_cells)  
        # Initializing Dataframe
        if (not ( self.dataframe is None))   and  ( self.reset_cell_counter == False): # IF the dataframe exist and not reset for multi-channel fish is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) +1
        elif (not ( self.dataframe is None)) and (self.reset_cell_counter == True):    # IF dataframe exist and reset is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) - n_masks +1   # restarting the counter for the number of cells
        else: # IF the dataframe does not exist.
            new_dataframe = pd.DataFrame( columns=['image_id', 'cell_id', 'spot_id','nucleus_y', 'nucleus_x','nuc_area_px','cyto_area_px', 'cell_area_px','z', 'y', 'x','is_nuc','is_cluster','cluster_size','spot_type','is_cell_fragmented'])
            counter_total_cells = 0
        # loop for each cell in image
        for id_cell in range (0,n_masks): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
            if not (self.channels_with_nucleus in  (None, [None])):
                nuc_area, nuc_centroid_y, nuc_centroid_x = mask_selector(self.masks_nuclei[id_cell], calculate_centroid=True)
                selected_mask_nuc = self.masks_nuclei[id_cell]
                tested_mask =  self.masks_nuclei[id_cell]
            else:
                nuc_area, nuc_centroid_y, nuc_centroid_x = 0,0,0
                selected_mask_nuc = None
            if not (self.channels_with_cytosol in (None, [None])) and not (self.channels_with_nucleus in  (None, [None])):
                cyto_area, _ ,_                          = mask_selector(self.masks_cytosol_no_nuclei[id_cell],calculate_centroid=False)
                slected_masks_cytosol_no_nuclei = self.masks_cytosol_no_nuclei[id_cell]
            else:
                cyto_area =0
                slected_masks_cytosol_no_nuclei = None
            if not (self.channels_with_cytosol in (None, [None])):
                cell_area, _, _                          = mask_selector(self.masks_complete_cells[id_cell],calculate_centroid=False)
                tested_mask =  self.masks_complete_cells[id_cell]
            else:
                cell_area = 0
            # case where no nucleus is passed and only cytosol
            if (self.channels_with_nucleus in  (None, [None])) and not (self.channels_with_cytosol in (None, [None])):
                slected_masks_cytosol_no_nuclei = self.masks_complete_cells[id_cell]
            
            is_cell_in_border =  np.any( np.concatenate( ( tested_mask[:,0],tested_mask[:,-1],tested_mask[0,:],tested_mask[-1,:] ) ) )   
            
            
            
            # Data extraction
            try:
                new_dataframe = data_to_df(new_dataframe, 
                                    self.spotDetectionCSV, 
                                    self.clusterDetectionCSV, 
                                    mask_nuc = selected_mask_nuc, 
                                    mask_cytosol_only=slected_masks_cytosol_no_nuclei, 
                                    nuc_area=nuc_area,
                                    cyto_area=cyto_area, 
                                    cell_area=cell_area, 
                                    centroid_y = nuc_centroid_y, 
                                    centroid_x = nuc_centroid_x,
                                    image_counter=self.image_counter,
                                    is_cell_in_border = is_cell_in_border,
                                    spot_type = self.spot_type ,
                                    cell_counter =counter_total_cells)
            except:
                print(selected_mask_nuc.max())
                print(slected_masks_cytosol_no_nuclei.max())
                print(new_dataframe)
                print('spots')
                print(self.spotDetectionCSV)
                print('cluster')
                print(self.clusterDetectionCSV)
            
            counter_total_cells +=1
        return new_dataframe


class SpotDetection():
    '''
    This class is intended to detect spots in FISH images using `Big-FISH <https://github.com/fish-quant/big-fish>`_. The format of the image must be  [Z, Y, X, C].
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter description obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
    For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    FISH_channels : int, or List
        List of channels with FISH spots that are used for the quantification
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    cluster_radius : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    masks_complete_cells : NumPy array
        Masks for every cell detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_nuclei: NumPy array
        Masks for every nucleus detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_cytosol_no_nuclei :  NumPy array
        Masks for every cytosol detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    dataframe : Pandas Dataframe 
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nucleus_y, nucleus_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    list_voxels : List of tupples or None
        list with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each FISH channel.
        voxel_size_z is the height of a voxel, along the z axis, in nanometers. The default is 300.
        voxel_size_yx is the size of a voxel on the yx plan in nanometers. The default is 150.
    list_psfs : List of tuples or None
        List with a tuple with two elements (psf_z, psf_yx ) for each FISH channel.
        psf_z is the size of the PSF emitted by a spot in the z plan, in nanometers. The default is 350.
        psf_yx is the size of the PSF emitted by a spot in the yx plan in nanometers.
    show_plots : bool, optional
        If True, it shows a 2D maximum projection of the image and the detected spots. The default is False.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is False.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calulated automatically.
    
    '''
    def __init__(self,image,  FISH_channels ,channels_with_cytosol,channels_with_nucleus, cluster_radius=350, minimum_spots_cluster=4, masks_complete_cells = None, masks_nuclei  = None, masks_cytosol_no_nuclei = None, dataframe=None,image_counter=0, list_voxels=[[500,200]], list_psfs=[[300,100]], show_plots=True,image_name=None,save_all_images=True,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None):
        self.image = image
        self.channels_with_cytosol=channels_with_cytosol
        self.channels_with_nucleus=channels_with_nucleus
        if not (masks_complete_cells is None) and (masks_nuclei is None):
            self.list_masks_complete_cells = Utilities().separate_masks(masks_complete_cells)
        else:
            self.list_masks_complete_cells = Utilities().separate_masks(masks_nuclei)
            
        if not (masks_nuclei is None):    
            self.list_masks_nuclei = Utilities().separate_masks(masks_nuclei)
        else:
            self.list_masks_nuclei = None
        
        if not (masks_complete_cells is None) and not (masks_nuclei is None):
            self.list_masks_cytosol_no_nuclei = Utilities().separate_masks(masks_cytosol_no_nuclei)
        else:
            self.list_masks_cytosol_no_nuclei = None
        self.FISH_channels = FISH_channels
        self.cluster_radius = cluster_radius
        self.minimum_spots_cluster = minimum_spots_cluster
        self.dataframe = dataframe
        self.image_counter = image_counter
        self.show_plots = show_plots
        if type(list_voxels[0]) != list:
            self.list_voxels = [list_voxels]
        else:
            self.list_voxels = list_voxels
        if type(list_psfs[0]) != list:
            self.list_psfs = [list_psfs]
        else:
            self.list_psfs = list_psfs
        # converting FISH channels to a list
        if not (type(FISH_channels) is list):
            self.list_FISH_channels = [FISH_channels]
        else:
            self.list_FISH_channels = FISH_channels
        self.image_name = image_name
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        self.threshold_for_spot_detection=threshold_for_spot_detection
        
    def get_dataframe(self):
        for i in range(0,len(self.list_FISH_channels)):
            print('Spot Detection for Channel :', str(self.list_FISH_channels[i]) )
            if (i ==0):
                dataframe_FISH = self.dataframe 
                reset_cell_counter = False
            voxel_size_z = self.list_voxels[i][0]
            voxel_size_yx = self.list_voxels[i][1]
            psf_z = self.list_psfs[i][0] 
            psf_yx = self.list_psfs[i][1]
            [spotDetectionCSV, clusterDetectionCSV] = BigFISH(self.image, self.list_FISH_channels[i], voxel_size_z = voxel_size_z,voxel_size_yx = voxel_size_yx, psf_z = psf_z, psf_yx = psf_yx, cluster_radius=self.cluster_radius,minimum_spots_cluster=self.minimum_spots_cluster, show_plots=self.show_plots,image_name=self.image_name,save_all_images=self.save_all_images,display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes,use_log_filter_for_spot_detection =self.use_log_filter_for_spot_detection,threshold_for_spot_detection=self.threshold_for_spot_detection).detect()
            dataframe_FISH = DataProcessing(spotDetectionCSV, clusterDetectionCSV, self.list_masks_complete_cells, self.list_masks_nuclei, self.list_masks_cytosol_no_nuclei, self.channels_with_cytosol,self.channels_with_nucleus, dataframe =dataframe_FISH,reset_cell_counter=reset_cell_counter,image_counter = self.image_counter ,spot_type=i).get_dataframe()
            # reset counter for image and cell number
            reset_cell_counter = True
        return dataframe_FISH
    
    
class Metadata():
    '''
    This class is intended to generate a metadata file containing used dependencies, user information, and parameters used to run the code.
    
    Parameters
    
    data_dir: str or PosixPath
        Directory containing the images to read.
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. 
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation.
    channels_with_FISH  : list of int
        List with integers indicating the index of channels for the FISH detection using.
    diameter_cytosol : int
        Average cytosol size in pixels. The default is 150.
    diameter_nucleus : int
        Average nucleus size in pixels. The default is 100.
    minimum_spots_cluster : int
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    list_voxels : List of lists or None
        List with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each FISH channel.
    list_psfs : List of lists or None
        List with a tuple with two elements (psf_z, psf_yx ) for each FISH channel.
    file_name_str : str
        Name used for the metadata file. The final name has the format metadata_<<file_name_str>>.txt
    '''
    def __init__(self,data_dir, channels_with_cytosol, channels_with_nucleus, channels_with_FISH, diameter_nucleus, diameter_cytosol, minimum_spots_cluster, list_voxels=None, list_psfs=None, file_name_str=None,list_segmentation_succesful=True):
        self.list_images, self.path_files, self.list_files_names, self.number_images = ReadImages(data_dir).read()
        self.channels_with_cytosol = channels_with_cytosol
        self.channels_with_nucleus = channels_with_nucleus
        if isinstance(channels_with_FISH, list): 
            self.channels_with_FISH = channels_with_FISH
        else:
            self.channels_with_FISH = [channels_with_FISH]
        self.diameter_nucleus = diameter_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.list_voxels = list_voxels
        self.list_psfs = list_psfs
        self.file_name_str=file_name_str
        self.minimum_spots_cluster = minimum_spots_cluster
        if  (not str(data_dir.name)[0:5] ==  'temp_') and (self.file_name_str is None):
            self.filename = './metadata_'+ str(data_dir.name) +'.txt'
        elif not(self.file_name_str is None):
            self.filename = './metadata_'+ str(file_name_str) +'.txt'
        else:
            self.filename = './metadata_'+ str(data_dir.name[5:]) +'.txt'
        self.data_dir = data_dir
        self.list_segmentation_succesful =list_segmentation_succesful
    def write_metadata(self):
        '''
        This method writes the metadata file.
        '''
        installed_modules = [str(module).replace(" ","==") for module in pkg_resources.working_set]
        important_modules = [ 'tqdm', 'torch','tifffile', 'setuptools', 'scipy', 'scikit-learn', 'scikit-image', 'PyYAML', 'pysmb', 'pyfiglet', 'pip', 'Pillow', 'pandas', 'opencv-python-headless', 'numpy', 'numba', 'natsort', 'mrc', 'matplotlib', 'llvmlite', 'jupyter-core', 'jupyter-client', 'joblib', 'ipython', 'ipython-genutils', 'ipykernel', 'cellpose', 'big-fish']
        def create_data_file(filename):
            if sys.platform == 'linux' or sys.platform == 'darwin':
                os.system('touch  ' + filename)
            elif sys.platform == 'win32':
                os.system('echo , > ' + filename)
        number_spaces_pound_sign = 75
        def write_data_in_file(filename):
            with open(filename, 'w') as fd:
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nAUTHOR INFORMATION  ')
                fd.write('\n    Author: ' + getpass.getuser())
                fd.write('\n    Created at: ' + datetime.datetime.today().strftime('%d %b %Y'))
                fd.write('\n    Time: ' + str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) )
                fd.write('\n    Operative System: ' + sys.platform )
                fd.write('\n    Hostname: ' + socket.gethostname() + '\n')
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nPARAMETERS USED  ')
                fd.write('\n    channels_with_cytosol: ' + str(self.channels_with_cytosol) )
                fd.write('\n    channels_with_nucleus: ' + str(self.channels_with_nucleus) )
                fd.write('\n    channels_with_FISH: ' + str(self.channels_with_FISH) )
                fd.write('\n    diameter_nucleus: ' + str(self.diameter_nucleus) )
                fd.write('\n    diameter_cytosol: ' + str(self.diameter_cytosol) )
                fd.write('\n    FISH parameters')
                for k in range (0,len(self.channels_with_FISH)):
                    fd.write('\n      For Channel ' + str(self.channels_with_FISH[k]) )
                    fd.write('\n        voxel_size_z: ' + str(self.list_voxels[k][0]) )
                    fd.write('\n        voxel_size_yx: ' + str(self.list_voxels[k][1]) )
                    fd.write('\n        psf_z: ' + str(self.list_psfs[k][0]) )
                    fd.write('\n        psf_yx: ' + str(self.list_psfs[k][1]) )
                fd.write('\n    minimum_spots_cluster: ' + str(self.minimum_spots_cluster) )
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nFILES AND DIRECTORIES USED ')
                fd.write('\n    Directory path: ' + str(self.data_dir) )
                fd.write('\n    Folder name: ' + str(self.data_dir.name)  )
                
                # for loop for all the images.
                fd.write('\n    Images in the directory :'  )
                for indx, img_name in enumerate (self.list_files_names):
                    if self.list_segmentation_succesful[indx]== True:
                        fd.write('\n        '+ img_name)
                    else:
                        fd.write('\n        '+ img_name + ' ===> image ignored for error during segmentation.')
                fd.write('\n')  
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nREPRODUCIBILITY ')
                fd.write('\n    Platform: \n')
                fd.write('        Python: ' + str(platform.python_version()) )
                fd.write('\n    Dependancies: ')
                # iterating for all modules
                for module_name in installed_modules:
                    if any(module_name[0:4] in s for s in important_modules):
                        fd.write('\n        '+ module_name)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
        create_data_file(self.filename)
        write_data_in_file(self.filename)
        return None


class PlotImages():
    '''
    This class intended to plot all the channels from an image with format  [Z, Y, X, C].
    
    Parameters
    
    image: NumPy array
        Array of images with dimensions [Z, Y, X, C].
    figsize : tuple with figure size, optional.
        Tuple with format (x_size, y_size). the default is (8.5, 5).
    '''
    def __init__(self,image,figsize=(8.5, 5),image_name='temp',show_plots=True):
        self.image = image
        self.figsize = figsize
        self.image_name = image_name
        self.show_plots = show_plots
        
    def plot(self):
        '''
        This method plots all the channels for the original image.
        '''
        print(self.image.shape)
        number_channels = self.image.shape[3]
        center_slice = self.image.shape[0]//2
        _, axes = plt.subplots(nrows=1, ncols=number_channels, figsize=self.figsize)
        for i in range (0,number_channels ):
            rescaled_image = RemoveExtrema(self.image[center_slice,:,:,i],min_percentile=1, max_percentile=98).remove_outliers() 
            img_2D = rescaled_image
            axes[i].imshow( img_2D ,cmap='Spectral') 
            axes[i].set_title('Channel_'+str(i))
            axes[i].grid(color='k', ls = '-.', lw = 0.5)
        plt.savefig(self.image_name,bbox_inches='tight')
        if self.show_plots ==True:
            plt.show()
        else:
            plt.close()
        return None


class ReportPDF():
    '''
    This class intended to create a PDF report including the images generated during the pipeline.
    
    Parameters
    
    directory_results: str or PosixPath
        Directory containing the images to include in the report.
    channels_with_FISH  : list of int
        List with integers indicating the index of channels for the FISH detection using.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is True.
    list_z_slices_per_image : int
        List containing all z-slices for each figure.
        
    .. image:: images/pdf_report.png
    
    This PDF file is generated, and it contains the processing steps for each image in the folder.
    
    '''    
    def __init__(self, directory, channels_with_FISH,save_all_images,list_z_slices_per_image,threshold_for_spot_detection,list_segmentation_succesful=True):
        self.directory = directory
        if isinstance(channels_with_FISH, list): 
            self.channels_with_FISH = channels_with_FISH
        else:
            self.channels_with_FISH = [channels_with_FISH]
        self.save_all_images=save_all_images
        self.list_z_slices_per_image = list_z_slices_per_image
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.list_segmentation_succesful =list_segmentation_succesful
    def create_report(self):
        #print(self.list_segmentation_succesful)
        '''
        This method creates a PDF with the original images, images for cell segmentation and images for the spot detection.
        '''
        pdf = FPDF()
        WIDTH = 210
        HEIGHT = 297
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        # code that reads the main file names
        list_files_names =[]
        substring_to_detect_in_file_name = 'ori_'               # prefix 'ori_' means original image 
        ending_string = re.compile(substring_to_detect_in_file_name)  # for faster search in the file
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file) and file[0]!= '.': # detecting a match in the end, not consider hidden files starting with '.'
                    file_name = file.rpartition('.')[0]         # getting the file name and removing extention
                    list_files_names.append(file_name[4:])      # removing the prefix from the file name.
        list_files_names.sort()
        # Main loop that reads each image and makes the pdf
        for i,temp_file_name in enumerate(list_files_names):
            #print(i, temp_file_name)
            pdf.cell(w=0, h=10, txt='Original image: ' + temp_file_name,ln =2,align = 'L')
            # code that returns the path of the original image
            temp_original_img_name = pathlib.Path().absolute().joinpath( self.directory, 'ori_' + temp_file_name +'.png' )
            pdf.image(str(temp_original_img_name), x=0, y=20, w=WIDTH-30)
            # creating some space
            for text_idx in range(0, 12):
                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
            pdf.cell(w=0, h=10, txt='Cell segmentation: ' + temp_file_name,ln =1,align = 'L')
            # code that returns the path of the segmented image
            if self.list_segmentation_succesful[i]==True:
                temp_segmented_img_name = pathlib.Path().absolute().joinpath( self.directory, 'seg_' + temp_file_name +'.png' )
                pdf.image(str(temp_segmented_img_name), x=0, y=HEIGHT/2, w=WIDTH-30)
            else:
                pdf.cell(w=0, h=20, txt='Segmentation was not possible for image: ' + temp_file_name,ln =1,align = 'L')
            
            # Code that plots the detected spots.
            if (self.save_all_images==True) and (self.list_segmentation_succesful[i]==True):
                for id_channel, channel in enumerate(self.channels_with_FISH):
                    counter=1
                    pdf.add_page() # adding a page
                    for z_slice in range(0, self.list_z_slices_per_image[i]):
                        temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel) + '_slice_'+ str(z_slice) +'.png' )
                        # Plotting bottom image
                        if counter%2==0: # Adding space if is an even counter
                            # adding some space to plot the bottom image
                            for j in range(0, 11):
                                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                            # Plotting the image
                            pdf.cell(w=0, h=0, txt='FISH Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L') 
                            pdf.image(str(temp_seg_name), x=0, y=HEIGHT//2, w=WIDTH-80)
                            pdf.add_page()
                        # plotting top image
                        else:
                            pdf.cell(w=0, h=10, txt='FISH Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L') 
                            pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-80)
                        counter=counter+1
                    pdf.add_page()
                    try:
                        if (self.threshold_for_spot_detection is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()
            elif self.list_segmentation_succesful[i]==True:
                pdf.add_page()
                for id_channel, channel in enumerate(self.channels_with_FISH):
                    # Plotting the image with detected spots
                    temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel)+'.png' )
                    pdf.cell(w=0, h=10, txt='FISH Ch_ ' + str(channel) + ': '+ temp_file_name,ln =2,align = 'L') 
                    pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-30)                    
                    # adding some space
                    for j in range(0, 12):
                        pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                    # Plotting the elbow plot
                    try:
                        if (self.threshold_for_spot_detection is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()                
                
        pdf_name =  'pdf_report_' + self.directory.name[13:] + '.pdf'
        pdf.output(pdf_name, 'F')
        return None


class PipelineFISH():
    '''
    This class is intended to perform complete FISH analyses including cell segmentation and spot detection.
    
    Parameters
    
    parameter: bool, optional
        parameter description. The default is True. 
    list_voxels : List of lists or None
        list with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each FISH channel.
    list_psfs : List of lists or None
        list with a tuple with two elements (psf_z, psf_yx ) for each FISH channel.
    list_masks : List of Numpy or None.
        list of Numpy arrays where each array has values from 0 to n where n is the number of masks in  the image.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is True.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calulated automatically.
    use_brute_force : bool, optional
        Flag to perform cell segmentation using all possible combination of parameters. The default is False.
    '''
    def __init__(self,data_dir, channels_with_cytosol=None, channels_with_nucleus=None, channels_with_FISH=None,diameter_nucleus=100, diameter_cytosol=200, minimum_spots_cluster=None,   masks_dir=None, show_plots=True,list_voxels=[[500,200]], list_psfs=[[300,100]],file_name_str =None,optimization_segmentation_method='z_slice_segmentation',save_all_images=True,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None,use_brute_force=False):
        list_images, self.path_files, self.list_files_names, self.number_images = ReadImages(data_dir).read()
        if len(list_images[0].shape) < 4:
            list_images_extended = [ np.expand_dims(img,axis=0) for img in list_images ] 
            self.list_images = list_images_extended
        else:
            self.list_images = list_images
        self.list_z_slices_per_image = [ img.shape[0] for img in self.list_images] # number of z-slices in the figure
        self.channels_with_cytosol = channels_with_cytosol
        self.channels_with_nucleus = channels_with_nucleus
        self.channels_with_FISH = channels_with_FISH
        self.diameter_nucleus = diameter_nucleus
        self.diameter_cytosol = diameter_cytosol
        if type(list_voxels[0]) != list:
            self.list_voxels = [list_voxels]
        else:
            self.list_voxels = list_voxels
        if type(list_psfs[0]) != list:
            self.list_psfs = [list_psfs]
        else:
            self.list_psfs = list_psfs
        self.minimum_spots_cluster = minimum_spots_cluster
        self.show_plots = show_plots
        self.CLUSTER_RADIUS = 500
        self.data_dir = data_dir
        if not(file_name_str is None):
            self.name_for_files = file_name_str
        else:
            self.name_for_files = self.data_dir.name
        self.masks_dir=masks_dir
        # saving the masks if they are not passed as a directory
        if (masks_dir is None):
            self.save_masks_as_file = True
        else:
            self.save_masks_as_file = False
        number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
        if number_gpus ==0:
            self.optimization_segmentation_method = None
        else:
            self.optimization_segmentation_method = optimization_segmentation_method # optimization_segmentation_method = 'intensity_segmentation' 'z_slice_segmentation', 'gaussian_filter_segmentation' , None
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        #if not (threshold_for_spot_detection is None):
        self.threshold_for_spot_detection=threshold_for_spot_detection
        
        self.use_brute_force =use_brute_force
        
    def run(self):
        
        NUMBER_OF_PIXELS_IN_MASK = 10000
        
        # Prealocating arrays
        list_masks_complete_cells=[]
        list_masks_nuclei=[]
        list_masks_cytosol_no_nuclei=[]
        list_segmentation_succesful=[]
        # temp_results_images
        temp_folder_name = str('temp_results_'+ self.name_for_files)
        if not os.path.exists(temp_folder_name):
            os.makedirs(temp_folder_name)
        if self.save_masks_as_file ==True:
            masks_folder_name = str('masks_'+ self.name_for_files)
            if not os.path.exists(masks_folder_name):
                os.makedirs(masks_folder_name)        
        # Running the pipeline.
        for i in range (0, self.number_images ):
            print( ' ############### ' )
            print( '       IMAGE : '+ str(i) )
            print( ' ############### \n ' )
            if i ==0:
                dataframe = None
            print('ORIGINAL IMAGE')
            print(self.list_files_names[i])
            temp_file_name = self.list_files_names[i][:self.list_files_names[i].rfind('.')] # slcing the name of the file. Removing after finding '.' in the string.
            temp_original_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'ori_' + temp_file_name +'.png' )
            PlotImages(self.list_images[i],figsize=(15, 10) ,image_name=  temp_original_img_name, show_plots = self.show_plots).plot()            
            # Cell segmentation
            temp_segmentation_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'seg_' + temp_file_name +'.png' )
            print('CELL SEGMENTATION')
            if (self.masks_dir is None):
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = CellSegmentation(self.list_images[i],self.channels_with_cytosol, self.channels_with_nucleus, diameter_cytosol = self.diameter_cytosol, diameter_nucleus=self.diameter_nucleus, show_plots=self.show_plots,optimization_segmentation_method = self.optimization_segmentation_method,image_name = temp_segmentation_img_name,use_brute_force=self.use_brute_force).calculate_masks() 
            # test if segmentation was succcesful
                if np.sum([masks_complete_cells.flatten(), masks_nuclei.flatten(), masks_cytosol_no_nuclei.flatten()]) > NUMBER_OF_PIXELS_IN_MASK:
                    segmentation_succesful = True
                else:
                    segmentation_succesful = False                
            else:
                segmentation_succesful = True
                # Paths to masks
                if not (self.channels_with_nucleus is None) and (segmentation_succesful==True) :
                    mask_nuc_path = self.masks_dir.absolute().joinpath('masks_nuclei_' + temp_file_name +'.tif' )
                    masks_nuclei = imread(str(mask_nuc_path)) 
                if not (self.channels_with_cytosol is None):
                    mask_cyto_path = self.masks_dir.absolute().joinpath( 'masks_cyto_' + temp_file_name +'.tif' )
                    masks_complete_cells = imread(str( mask_cyto_path   )) 
                if not (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None) and (segmentation_succesful==True) :
                    mask_cyto_no_nuclei_path = self.masks_dir.absolute().joinpath('masks_cyto_no_nuclei_' + temp_file_name +'.tif' )
                    masks_cytosol_no_nuclei = imread(str(mask_cyto_no_nuclei_path  ))
                # Plotting masks
                if not (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None) and (segmentation_succesful==True) :
                    _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 10))
                    axes[0].imshow(masks_complete_cells)
                    axes[0].set(title = 'Cytosol mask')
                    axes[1].imshow(masks_nuclei)
                    axes[1].set(title = 'Nuclei mask')
                    axes[2].imshow(masks_cytosol_no_nuclei)
                    axes[2].set(title = 'Cytosol only')
                if not (self.channels_with_cytosol is None) and (self.channels_with_nucleus is None) and (segmentation_succesful==True) :
                    _, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 10))
                    axes[0].imshow(masks_complete_cells)
                    axes[0].set(title = 'Cytosol mask')
                if (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None) and (segmentation_succesful==True) :
                    _, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 10))
                    axes[0].imshow(masks_nuclei)
                    axes[0].set(title = 'Nuclei mask')
                plt.savefig(temp_segmentation_img_name,bbox_inches='tight')
                if self.show_plots == True:
                    plt.show()
                else:
                    plt.close()
                
            # saving masks
            if (self.save_masks_as_file ==True) and (segmentation_succesful==True) :
                if not (self.channels_with_nucleus is None):
                    mask_nuc_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_nuclei_' + temp_file_name +'.tif' )
                    tifffile.imwrite(mask_nuc_path, masks_nuclei)
                if not (self.channels_with_cytosol is None):
                    mask_cyto_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_cyto_' + temp_file_name +'.tif' )
                    tifffile.imwrite(mask_cyto_path, masks_complete_cells)
                if not (self.channels_with_cytosol is None) and not (self.channels_with_nucleus is None):
                    mask_cyto_no_nuclei_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_cyto_no_nuclei_' + temp_file_name +'.tif' )
                    tifffile.imwrite(mask_cyto_no_nuclei_path, masks_cytosol_no_nuclei)
            
            print('SPOT DETECTION')
            if segmentation_succesful==True:
                temp_detection_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'det_' + temp_file_name )
                dataframe_FISH = SpotDetection(self.list_images[i],self.channels_with_FISH,self.channels_with_cytosol,self.channels_with_nucleus,cluster_radius=self.CLUSTER_RADIUS,minimum_spots_cluster=self.minimum_spots_cluster,masks_complete_cells=masks_complete_cells, masks_nuclei=masks_nuclei, masks_cytosol_no_nuclei=masks_cytosol_no_nuclei, dataframe=dataframe,image_counter=i, list_voxels=self.list_voxels,list_psfs=self.list_psfs, show_plots=self.show_plots,image_name = temp_detection_img_name,save_all_images=self.save_all_images,display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes,use_log_filter_for_spot_detection=self.use_log_filter_for_spot_detection,threshold_for_spot_detection=self.threshold_for_spot_detection).get_dataframe()
                dataframe = dataframe_FISH
                list_masks_complete_cells.append(masks_complete_cells)
                list_masks_nuclei.append(masks_nuclei)
                list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                del masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei
            # appending cell segmentation flag
            list_segmentation_succesful.append(segmentation_succesful)
        
        # Creating the dataframe       
        if  (not str(self.name_for_files)[0:5] ==  'temp_') and np.sum(list_segmentation_succesful)>0:
            dataframe.to_csv('dataframe_' + self.name_for_files +'.csv')
        elif np.sum(list_segmentation_succesful)>0:
            dataframe.to_csv('dataframe_' + self.name_for_files[5:] +'.csv')
        # Creating the metadata
        Metadata(self.data_dir, self.channels_with_cytosol, self.channels_with_nucleus, self.channels_with_FISH,self.diameter_nucleus, self.diameter_cytosol, self.minimum_spots_cluster,list_voxels=self.list_voxels, list_psfs=self.list_psfs,file_name_str=self.name_for_files,list_segmentation_succesful=list_segmentation_succesful).write_metadata()
        # Creating a PDF report
        ReportPDF(directory=pathlib.Path().absolute().joinpath(temp_folder_name) , channels_with_FISH=self.channels_with_FISH, save_all_images=self.save_all_images, list_z_slices_per_image=self.list_z_slices_per_image,threshold_for_spot_detection=self.threshold_for_spot_detection,list_segmentation_succesful=list_segmentation_succesful ).create_report()
        return dataframe, list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei
