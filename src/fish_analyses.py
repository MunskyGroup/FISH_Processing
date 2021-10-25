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
    import smb 
    import glob
    import tifffile
    # importing tensor flow
    import os; from os import listdir; from os.path import isfile, join
    # to remove tensorflow warnings
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #import tensorflow as tf
    #tf.get_logger().setLevel('ERROR')
    #from tensorflow.python.util import deprecation
    #deprecation._PRINT_DEPRECATION_WARNINGS = False
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    import sys
    from cellpose import models
    # Skimage
    from skimage import img_as_float64, img_as_uint
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_minimum
    from skimage.morphology import binary_closing
    from skimage.measure import find_contours
    from skimage.draw import polygon
    from skimage.util import random_noise
    from skimage.transform import warp
    from skimage import transform
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_laplace
    from skimage.draw import polygon_perimeter
    from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
    from skimage.morphology import square, dilation
    from skimage.io import imread

    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib.path as mpltPath
    from matplotlib import gridspec

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
    def connect():
        pass
    def close():
        pass
    def read():
        pass
    def write():
        pass
    
class ReadImages():
    '''
    This class reads all .tif images in a given folder and returns the names of these files, path, and number of files.
    
    Parameters
    --  --  --  --  -- 

    directory: str or PosixPath
        Directory containing the images to merge.
    '''    
    def __init__(self, directory:str ):
        if type(directory)== pathlib.PosixPath:
            self.directory = directory
        else:
            self.directory = Path(directory)
    def read(self):
        '''
        Method takes all the videos in the folder and merge those with similar names.

        Returns
        --  --  -- -
        list_images : List of NumPy arrays. 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C] or [T, Y, X, C] . 
        list_file_names : List of strings 
            List with strings of names.
        number_files : int. 
            Number of images in the folder.
        '''
        list_files_names = sorted([f for f in listdir(self.directory) if isfile(join(self.directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files = [ str(self.directory.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file
        number_files = len(path_files)
        list_images = [imread(f) for f in path_files]
        return list_images, path_files, list_files_names, number_files

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



class Cellpose():
    '''
    This class is intended to detect cells by image masking using **Cellpose** . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.

    Parameters
    --  --  --  --  -- 
    video : NumPy array
        Array of images with dimensions [T, Y, X, C].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0, 0], for RGB images with intensity for cytosol and nuclei use [0, 1] . The default is [0, 0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        To detect between the two options: 'cyto' or 'nuclei'. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells' or 'max_cells_and_area'. The default is 'max_cells_and_area'.
    '''
    def __init__(self, video:np.ndarray, num_iterations:int = 5, channels:list = [0, 0], diameter:float = 120, model_type:str = 'cyto', selection_method:str = 'max_cells_and_area'):
        self.video = video
        self.num_iterations = num_iterations
        self.minimumm_probability = 0
        self.maximum_probability = 4
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
        #self.NUMBER_OF_CORES = 1
        #self.range_diameter = np.round(np.linspace(100, 300, num_iterations), 0)
        self.tested_probabilities = np.round(np.linspace(self.minimumm_probability, self.maximum_probability, self.num_iterations), 2)

    def calculate_masks(self):
        '''
        This method performs the process of image masking using **Cellpose**.

        Returns
        --  --  -- -
        selected_masks : List of NumPy arrays
            List of NumPy arrays with values between 0 and the number of detected cells in the image, where a number larger than zero represents the masked area for each cell, and 0 represents the area where no cells are detected.
        '''
        # Next two lines suppressing output from cellpose
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        model = models.Cellpose(gpu = 1, model_type = self.model_type) # model_type = 'cyto' or model_type = 'nuclei'
        # Loop that test multiple probabilities in cell pose and returns the masks with the longest area.
        
        #area_longest_mask = np.zeros(self.num_iterations)

        def cellpose_max_area( cell_prob):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = cell_prob, diameter = self.diameter, min_size = -1, channels = self.channels, progress = None)
            except:
                masks =0
            n_masks = np.amax(masks)
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
        
        def cellpose_max_cells(cell_prob):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = cell_prob, diameter =self.diameter, min_size = -1, channels = self.channels, progress = None)
            except:
                masks =0
            return np.amax(masks)

        def cellpose_max_cells_and_area( cell_prob):
            try:
                masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = cell_prob, diameter = self.diameter, min_size = -1, channels = self.channels, progress = None)
            except:
                masks =0
            n_masks = np.amax(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    size_mask.append(np.sum(masks == nm)) # creating a list with the size of each mask
                number_masks= np.amax(masks)
                metric = np.sum(np.asarray(size_mask)) * number_masks
                return metric
            if n_masks == 1: # do nothing if only a single mask is detected per image.
                return np.sum(masks == 1)
            else:  # return zero if no mask are detected
                return 0     

        if self.selection_method == 'max_area':
            #list_metrics_masks = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(cellpose_max_area)(tested_probabilities[i] ) for i,_ in enumerate(tested_probabilities))
            list_metrics_masks = [cellpose_max_area(self.tested_probabilities[i],  ) for i,_ in enumerate(self.tested_probabilities)]
            #evaluated_metric_for_masks = np.asarray([list_metrics_masks[i]  for i,_ in enumerate(self.tested_probabilities)]   )
            #list_metrics_masks = [[cellpose_max_area(self.tested_probabilities[i],self.range_diameter[j] ) for i,_ in enumerate(self.tested_probabilities)] for j,_ in enumerate(self.range_diameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)

        if self.selection_method == 'max_cells':
            #list_metrics_masks = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(cellpose_max_cells)(tested_probabilities[i] ) for i,_ in enumerate(tested_probabilities))
            list_metrics_masks = [cellpose_max_cells(self.tested_probabilities[i] ) for i,_ in enumerate(self.tested_probabilities)]
            #evaluated_metric_for_masks = np.asarray([list_metrics_masks[i]  for i,_ in enumerate(self.tested_probabilities)]   )
            #list_metrics_masks = [[cellpose_max_cells(self.tested_probabilities[i],self.range_diameter[j] ) for i,_ in enumerate(self.tested_probabilities)] for j,_ in enumerate(self.range_diameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)

        if self.selection_method == 'max_cells_and_area':
            list_metrics_masks = [cellpose_max_cells_and_area(self.tested_probabilities[i] ) for i,_ in enumerate(self.tested_probabilities)]
            #evaluated_metric_for_masks = np.asarray([list_metrics_masks[i]  for i,_ in enumerate(self.tested_probabilities)]   )
            #list_metrics_masks = [[cellpose_max_cells_and_area(self.tested_probabilities[i],self.range_diameter[j] ) for i,_ in enumerate(self.tested_probabilities)] for j,_ in enumerate(self.range_diameter)]
            evaluated_metric_for_masks = np.asarray(list_metrics_masks)

   
        if np.amax(evaluated_metric_for_masks) >0:
            selected_conditions = self.tested_probabilities[np.argmax(evaluated_metric_for_masks)]
            #selected_conditions = np.argwhere(evaluated_metric_for_masks==evaluated_metric_for_masks.max())
            selected_masks, _, _, _ = model.eval(self.video, normalize = True, cellprob_threshold = selected_conditions, diameter = self.diameter, min_size = -1, channels = self.channels, progress = None)
            #selected_masks[0, :] = 0;selected_masks[:, 0] = 0;selected_masks[selected_masks.shape[0]-1, :] = 0;selected_masks[:, selected_masks.shape[1]-1] = 0#This line of code ensures that the corners are zeros.
            #selected_masks[0:10, :] = 0;selected_masks[:, 0:10] = 0;selected_masks[selected_masks.shape[0]-10:selected_masks.shape[0]-1, :] = 0; selected_masks[:, selected_masks.shape[1]-10: selected_masks.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.

        else:
            selected_masks = None
            print('No cells detected on the image')

        sys.stdout.close()
        sys.stdout = old_stdout
        return selected_masks


class CellposeFISH():
    '''
    This class is intended to detect cells in FISH images using **Cellpose**. The class uses optimization to generate the meta-parameters used by cellpose. This class segments the nucleus and cytosol for every cell detected in the image.

    Parameters
    --  --  --  --  -- 
    video : NumPy array
        Array of images with dimensions [Z, Y, X, C] or maximum projection with dimensions [Y,X,C].
    channel_with_cytosol : List of int or None, optional
        DESCRIPTION. The default is None.
    channel_with_nucleus : list of int or None, optional
        DESCRIPTION. The default is None.
    selected_z_slice : int, optional
        DESCRIPTION. The default is 5.
    diameter_cytosol : float, optional
        Average cytosol size in pixels. The default is 150.
    diamter_nucleus : float, optional
        Average nucleus size in pixels. The default is 100.
    remove_fragmented_cells: bool, optional
        If true, it removes masks  in the border of the image. The default is 0.
    show_plot : bool, optional
        If true, it shows a plot with the detected masks. The default is 1.
    '''
    def __init__(self, video:np.ndarray, channel_with_cytosol = None, channel_with_nucleus= None, selected_z_slice:int = 5, diameter_cytosol:float = 150, diamter_nucleus:float = 100, remove_fragmented_cells:bool=0, show_plot: bool = 1):
        self.video = video
        self.selected_z_slice = selected_z_slice
        self.channel_with_cytosol = channel_with_cytosol
        self.channel_with_nucleus = channel_with_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.diamter_nucleus = diamter_nucleus
        self.show_plot = show_plot
        NUMBER_TESTED_THRESHOLDS = 5
        self.tested_thresholds = np.round(np.linspace(0, 3, NUMBER_TESTED_THRESHOLDS), 0)
        self.remove_fragmented_cells = remove_fragmented_cells
    def calculate_masks(self):
        '''
        This method performs the process of cell detection for FISH images using **Cellpose**.

        Returns
        --  --  -- -
        list_masks_complete_cells : List of NumPy arrays or a single NumPy array
            Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        list_masks_nuclei : List of NumPy arrays or a single NumPy array
            Masks for the nuclei for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        list_masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array
            Masks for every nucleus and cytosol for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
        index_paired_masks: List of pairs of int
            List of pairs of integers that associates the detected nuclei and cytosol.
        '''
        # This function is intended to separate masks in list of submasks
        def separate_masks (masks):
            list_masks = []
            n_masks = np.amax(masks)
            if not ( n_masks is None):
                if n_masks > 1: # detecting if more than 1 mask are detected per cell
                    #number_particles = []
                    for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                        mask_copy = masks.copy()
                        tested_mask = np.where(mask_copy == nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                        list_masks.append(tested_mask)
                else:  # do nothing if only a single mask is detected per image.
                    list_masks.append(masks)
            else:
                list_masks.append(masks)
            return list_masks
        # function that determines if a cell is in the border of the image
        def remove_fragmented(img_masks):
            img_masks_copy = np.copy(img_masks)
            for nm in range(1,np.amax(img_masks)+1):
                tested = np.where(img_masks_copy == nm, 1, 0)   # making zeros all elements outside each mask, and once all elements inside of each mask.
                # testing if tested is touching the border of the image
                is_border = np.any( np.concatenate( ( tested[:,0],tested[:,-1],tested[0,:],tested[-1,:] ) ) )
                if is_border == True:
                    img_masks = np.where(img_masks == nm, 0, img_masks)
            return img_masks
        
        # function that determines if the nucleus is in the cytosol
        def is_nucleus_in_cytosol(mask_nucleus, mask_cyto):
            nucleusInCell = []
            nucleusInCell.append(0) # Appending an empty value, to avoid errors
            contuour_n = find_contours(mask_nucleus, 0.5)
            contuour_c = find_contours(mask_cyto, 0.5)
            path = mpltPath.Path(contuour_c[0])
            # calculating the center of the mask applying a 10% reduction
            x_coord = [[i][0][1] for i in contuour_n[0]]
            y_coord = [[i][0][0] for i in contuour_n[0]]
            # Center of mask
            center_value = 0.5
            x_center = center_value* min(x_coord) + center_value * max(x_coord)
            y_center = center_value* min(y_coord) + center_value* max(y_coord)
            test_point = [y_center, x_center]
            if path.contains_point(test_point) == 1:
                return 1
            else:
                return 0
        # This function takes two list of images for masks and returns a list of lists with pairs indicating the nucleus masks that are contained in a cytosol
        def paired_masks(list_masks_nuclei:list, list_masks_cyto:list):
            n_masks_nuclei = len (list_masks_nuclei)
            n_masks_cyto = len(list_masks_cyto)
            array_paired_masks = np.zeros((n_masks_cyto, n_masks_nuclei)) # This array has dimensions n_masks_cyto, n_masks_nuclei and each entry indicate if the masks are paired
            for i in range(0, n_masks_cyto):
                for j in range (0, n_masks_nuclei):
                    try:
                        array_paired_masks[i, j] = is_nucleus_in_cytosol(list_masks_nuclei[j], list_masks_cyto[i])
                    except:
                        array_paired_masks[i, j] = 0
            # vertical array with paired masks
            itemindex = np.where(array_paired_masks == 1)
            index_paired_masks = np.vstack((itemindex[0], itemindex[1])).T
            return index_paired_masks # This array has dimensions [n_masks_cyto, n_masks_nuclei]
        # This function creates a mask for each cell.
        def generate_masks_complete_cell(index_paired_masks:np.ndarray, list_separated_masks_cyto:list):
            list_masks_complete_cells = []
            #list_idx = []
            for i in range(0, index_paired_masks.shape[0]):
                sel_mask_c = index_paired_masks[i][0]
                #if sel_mask_c not in list_idx: # conditional expersion to check if the mask is not duplicated
                    #list_idx.append(index_paired_masks[i][0]) # creating a list of indexes of masks to avoid saving the same mask two times
                list_masks_complete_cells.append(list_separated_masks_cyto[sel_mask_c])
            return list_masks_complete_cells
        # This function creates a mask for each nuclei
        def generate_masks_nuclei(index_paired_masks:np.ndarray, list_separated_masks_nuclei:list):
            list_masks_nuclei = []
            for i in range(0, index_paired_masks.shape[0]):
                sel_mask_n = index_paired_masks[i][1]
                list_masks_nuclei.append(list_separated_masks_nuclei[sel_mask_n])
            return list_masks_nuclei
        # This function creates a mask for each cytosol without nucleus
        def generate_masks_cytosol_no_nuclei(index_paired_masks:np.ndarray, list_masks_complete_cells:list, list_masks_nuclei:list):
            list_masks_cytosol_no_nuclei = []
            for i in range(0, index_paired_masks.shape[0]):
                substraction = list_masks_complete_cells[i] - list_masks_nuclei[i]
                substraction[substraction < 0] = 0
                list_masks_cytosol_no_nuclei.append(substraction)
            return list_masks_cytosol_no_nuclei
        def join_nulcei_masks(index_paired_masks:np.ndarray, list_masks_nuclei:list):
            # this code detects duplicated mask for the nucleus in the same cell and replaces with a joined mask. Also deletes from the list the duplicated elements.
            index_masks_cytosol = index_paired_masks[:, 0]
            idxs_to_delete = []
            for i in range(0, len (index_masks_cytosol)):
                duplicated_masks_idx = np.where(index_masks_cytosol == i)[0]
                if len(duplicated_masks_idx) > 1:
                    joined_mask = np.zeros_like(list_masks_nuclei[0])
                    for j in range(0, len(duplicated_masks_idx)):
                        joined_mask = joined_mask + list_masks_nuclei[duplicated_masks_idx[j]]
                        joined_mask[joined_mask > 0] = 1
                    list_masks_nuclei[duplicated_masks_idx[0]] = joined_mask # replacing the first duplication occurrence with the joined mask
                    idxs_to_delete.append(duplicated_masks_idx[1::][0])
            # creating a new list with the joined masks
            list_mask_joined = []
            for i in range(0, len (list_masks_nuclei)):
                if i not in idxs_to_delete:
                    list_mask_joined.append(list_masks_nuclei[i])
            # removing from index
            new_index_paired_masks = np.delete(index_paired_masks, idxs_to_delete, axis = 0)
            return list_mask_joined, new_index_paired_masks
        
        ##### IMPLEMENTATION #####
        if len(self.video.shape) > 3:  # [ZYXC]
            video_normalized = np.mean(self.video[2:-2,:,:,:],axis=0)    # taking the mean value
        else:
            video_normalized = self.video # [YXC]       
        
        def function_to_find_masks (video):
            # Cellpose
            #try:
            if not (self.channel_with_cytosol is None):
                #masks_cyto = Cellpose(video, channels = self.channel_with_cytosol,diameter = self.diameter_cytosol, model_type = 'cyto', selection_method = 'max_cells_and_area' ).calculate_masks()
                masks_cyto = Cellpose(video[:, :, self.channel_with_cytosol],diameter = self.diameter_cytosol, model_type = 'cyto', selection_method = 'max_cells_and_area' ).calculate_masks()
                if self.remove_fragmented_cells ==1:
                    masks_cyto= remove_fragmented(masks_cyto)
            
            if not (self.channel_with_nucleus is None):
                #masks_nuclei = Cellpose(video, channels = self.channel_with_nucleus,  diameter = self.diamter_nucleus, model_type = 'nuclei', selection_method = 'max_cells_and_area').calculate_masks()
                masks_nuclei = Cellpose(video[:, :, self.channel_with_nucleus],  diameter = self.diamter_nucleus, model_type = 'nuclei', selection_method = 'max_cells_and_area').calculate_masks()
                if self.remove_fragmented_cells ==1:
                    masks_nuclei= remove_fragmented(masks_nuclei)
            
            #except:
            #    masks_cyto = None
            #    masks_nuclei = None
            if not (self.channel_with_cytosol is None) and not(self.channel_with_nucleus is None):
                # Implementation
                list_separated_masks_nuclei = separate_masks(masks_nuclei)
                list_separated_masks_cyto = separate_masks(masks_cyto)
                # Array with paired masks
                index_paired_masks  =  paired_masks(list_separated_masks_nuclei, list_separated_masks_cyto)
                # Optional section that joins multiple nucleus masks
                id_c = index_paired_masks[:, 0].tolist()
                duplicated_nuclei_in_masks = any(id_c.count(x) > 1 for x in id_c)
                if duplicated_nuclei_in_masks == True:
                    list_masks_nuclei, index_paired_masks = join_nulcei_masks(index_paired_masks, list_separated_masks_nuclei)
                # List of mask
                list_masks_complete_cells = generate_masks_complete_cell(index_paired_masks, list_separated_masks_cyto)
                list_masks_nuclei = generate_masks_nuclei(index_paired_masks, list_separated_masks_nuclei)
                list_masks_cytosol_no_nuclei = generate_masks_cytosol_no_nuclei(index_paired_masks, list_masks_complete_cells, list_masks_nuclei)
            else:
                if not (self.channel_with_cytosol is None):
                    list_masks_complete_cells = separate_masks(masks_cyto) # []
                    list_masks_nuclei = []
                    list_masks_cytosol_no_nuclei = []
                    index_paired_masks =[]
                    masks_cyto = None
                    masks_nuclei= None
                if not (self.channel_with_nucleus is None):
                    list_masks_complete_cells = []
                    list_masks_nuclei = separate_masks(masks_nuclei)
                    list_masks_cytosol_no_nuclei = []
                    index_paired_masks =[]
                    masks_cyto = None
            return list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, index_paired_masks, masks_cyto, masks_nuclei

        # Section of the code that optimizes to find the maximum number of index_paired_masks
        if not (self.channel_with_cytosol is None) and not(self.channel_with_nucleus is None):
            list_sotring_number_paired_masks = []
            for idx, threshold in enumerate(self.tested_thresholds):
                video_copy = video_normalized.copy()
                video_temp = RemoveExtrema(video_copy,min_percentile=threshold, max_percentile=100-threshold,selected_channels=self.channel_with_cytosol).remove_outliers() 
                list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, index_paired_masks, masks_cyto,masks_nuclei = function_to_find_masks (video_temp)
                list_sotring_number_paired_masks.append(len(list_masks_cytosol_no_nuclei))
            array_number_paired_masks = np.asarray(list_sotring_number_paired_masks)
            print('arr',array_number_paired_masks)
            print('amax',np.argmax(array_number_paired_masks))
            selected_threshold = self.tested_thresholds[np.argmax(array_number_paired_masks)]
            print('sel',selected_threshold)
        else:
            selected_threshold = 0
        # Running the mask selection once a threshold is obtained
        video_copy = video_normalized.copy()
        video_temp = RemoveExtrema(video_copy,min_percentile=selected_threshold,max_percentile=100-selected_threshold,selected_channels=self.channel_with_cytosol).remove_outliers() 
        list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, index_paired_masks, masks_cyto,masks_nuclei  = function_to_find_masks (video_temp)

        if len(index_paired_masks) != 0 and not(self.channel_with_cytosol is None) and not(self.channel_with_nucleus is None):
            if self.show_plot == 1:
                #number_channels= self.video.shape[-1]
                _, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 10))
                im = video_normalized[ :, :, 0:3].copy()
                imin, imax = np.min(im), np.max(im); im -= imin
                imf = np.array(im, 'float32')
                imf *= 255./(imax-imin)
                im = np.asarray(np.round(imf), 'uint8')
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_cyto)
                axes[1].set(title = 'Cytosol mask')
                axes[2].imshow(masks_nuclei)
                axes[2].set(title = 'Nuclei mask')
                axes[3].imshow(im)
                for i in range(0, index_paired_masks.shape[0]):
                    contuour_n = find_contours(list_masks_nuclei[i], 0.5)
                    contuour_c = find_contours(list_masks_complete_cells[i], 0.5)
                    axes[3].fill(contuour_n[0][:, 1], contuour_n[0][:, 0], facecolor = 'none', edgecolor = 'yellow') # mask nucleus
                    axes[3].fill(contuour_c[0][:, 1], contuour_c[0][:, 0], facecolor = 'none', edgecolor = 'yellow') # mask cytosol
                    axes[3].set(title = 'Paired masks')
                plt.show()
        else:
            if not(self.channel_with_cytosol is None) and (self.channel_with_nucleus is None):
                if self.show_plot == 1:
                    #number_channels= self.video.shape[-1]
                    _, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
                    im = video_normalized[ :, :, 0:3].copy()
                    imin, imax = np.min(im), np.max(im); im -= imin
                    imf = np.array(im, 'float32')
                    imf *= 255./(imax-imin)
                    im = np.asarray(np.round(imf), 'uint8')
                    axes[0].imshow(im)
                    axes[0].set(title = 'All channels')
                    axes[1].imshow(masks_cyto)
                    axes[1].set(title = 'Cytosol mask')
                    #axes[2].imshow(im)
                    #for i in range(0, index_paired_masks.shape[0]):
                    #    contuour_c = find_contours(list_masks_complete_cells[i], 0.5)
                    #    axes[2].fill(contuour_c[0][:, 1], contuour_c[0][:, 0], facecolor = 'none', edgecolor = 'yellow') # mask cytosol
                    #    axes[2].set(title = 'Paired masks')
                    plt.show()

            if (self.channel_with_cytosol is None) and not(self.channel_with_nucleus is None):
                if self.show_plot == 1:
                    #number_channels= self.video.shape[-1]
                    _, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
                    im = video_normalized[ :, :, 0:3].copy()
                    imin, imax = np.min(im), np.max(im); im -= imin
                    imf = np.array(im, 'float32')
                    imf *= 255./(imax-imin)
                    im = np.asarray(np.round(imf), 'uint8')
                    axes[0].imshow(im)
                    axes[0].set(title = 'All channels')
                    axes[1].imshow(masks_nuclei)
                    axes[1].set(title = 'Nuclei mask')
                    #axes[2].imshow(im)
                    #for i in range(0, index_paired_masks.shape[0]):
                    #    contuour_n = find_contours(list_masks_nuclei[i], 0.5)
                    #    axes[2].fill(contuour_n[0][:, 1], contuour_n[0][:, 0], facecolor = 'none', edgecolor = 'yellow') # mask nucleus
                    #    axes[2].set(title = 'Paired masks')
                    plt.show()
            print('No paired masks were detected for this image')

        if (self.channel_with_cytosol is None):
            index_paired_masks = np.linspace(0, len(list_masks_nuclei)-1, len(list_masks_nuclei), dtype='int32')

        if (self.channel_with_nucleus is None):
            index_paired_masks = np.linspace(0, len(list_masks_complete_cells)-1, len(list_masks_complete_cells), dtype='int32')
        return list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, index_paired_masks



class RemoveExtrema():
    '''
    This class is intended to remove extreme values from a video. The format of the video must be [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].

    Parameters
    --  --  --  --  -- 
    video : NumPy array
        Array of images with dimensions [Y, X] , [Y, X, C] , [Z, Y, X, C] or [T, Y, X, C].
    min_percentile : float, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : float, optional
        Higher bound to normalize intensity. The default is 99.
    selected_channels : List or None, optional
        Use this option to select a list channels to remove extrema. The default is None and applies the removal of extrema to all the channels.
    '''
    def __init__(self, video:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels = None):
        self.video = video
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        #self.ignore_channel = ignore_channel
        if not (type(selected_channels) is list):
                self.selected_channels = [selected_channels]
        else:
            self.selected_channels =selected_channels

    def remove_outliers(self):
        '''
        This method normalizes the values of a video by removing extreme values.

        Returns
        --  --  -- -
        normalized_video : np.uint16
            Normalized video. Array with dimensions [T, Y, X, C] or image with format [Y, X].
        '''
        normalized_video = np.copy(self.video)
        normalized_video = np.array(normalized_video, 'float32');
        # Normalization code for image with format [Y, X]
        if len(self.video.shape) == 2:
            number_timepoints = 1
            number_channels = 1
            normalized_video_temp = normalized_video
            if not np.amax(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                max_val = np.percentile(normalized_video_temp, self.max_percentile)
                min_val = np.percentile(normalized_video_temp, self.min_percentile)
                normalized_video_temp [normalized_video_temp > max_val] = max_val
                normalized_video_temp [normalized_video_temp < min_val] = min_val
                normalized_video_temp [normalized_video_temp < 0] = 0
        
        # Normalization for video with format [Y, X, C].
        if len(self.video.shape) == 3:
            number_channels   = self.video.shape[2]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    normalized_video_temp = normalized_video[ :, :, index_channels]
                    if not np.amax(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                        max_val = np.percentile(normalized_video_temp, self.max_percentile)
                        min_val = np.percentile(normalized_video_temp, self.min_percentile)
                        normalized_video_temp [normalized_video_temp > max_val] = max_val
                        normalized_video_temp [normalized_video_temp < min_val] =  min_val
                        normalized_video_temp [normalized_video_temp < 0] = 0
        
        # Normalization for video with format [T, Y, X, C] or [Z, Y, X, C].
        if len(self.video.shape) == 4:
            number_timepoints, number_channels   = self.video.shape[0], self.video.shape[3]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                #if not self.ignore_channel == index_channels:
                    for index_time in range (number_timepoints):
                        normalized_video_temp = normalized_video[index_time, :, :, index_channels]
                        if not np.amax(normalized_video_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                            max_val = np.percentile(normalized_video_temp, self.max_percentile)
                            min_val = np.percentile(normalized_video_temp, self.min_percentile)
                            normalized_video_temp [normalized_video_temp > max_val] = max_val
                            normalized_video_temp [normalized_video_temp < min_val] = min_val
                            normalized_video_temp [normalized_video_temp < 0] = 0
        return np.asarray(normalized_video, 'uint16')

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
    def detect():
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
    
    
class Metadata():
    '''
    Description for the class.
    
    Parameters
    --  --  --  --  -- 

    parameter: bool, optional
        parameter description. The default is True. 
    '''
    def __init__(self,argument):
        pass
    


