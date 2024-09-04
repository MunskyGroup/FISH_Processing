import bigfish.stack as stack
import pandas as pd
import numpy as np
import glob
import tifffile
import os;
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import pathlib
import shutil
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
from  matplotlib.ticker import FuncFormatter

font_props = {'size': 16}

from src.Util.ReadImages import  ReadImages
from src.Util.NASConnection import NASConnection
from src.Util.MergeChannels import MergeChannels
from src.Util.RemoveExtrema import RemoveExtrema

# from Util import RemoveExtrema
# from .ReadImages import ReadImages
# from .NASConnection import NASConnection
# from .MergeChannels import MergeChannels




class Utilities():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass
    def remove_images_not_processed(images_metadata, list_images):
        if images_metadata is None:
            return list_images
        else:
            selected_images = []
            max_image_id = images_metadata['Image_id'].max()+1
            for i in range (max_image_id):
                processing_status = images_metadata[images_metadata['Image_id'] == i].Processing.values[0]
                if processing_status == 'successful':
                    selected_images.append(list_images[i])
        return selected_images
    
    def calculate_sharpness(self,list_images, channels_with_FISH, neighborhood_size=31, threshold=1.12):
        list_mean_sharpeness_image = []
        list_is_image_sharp=[]
        list_sharp_images =[]
        for _ , image in enumerate(list_images):
            temp = image[:,:,:,channels_with_FISH[0]].astype(np.uint16)
            focus = stack.compute_focus(temp, neighborhood_size=neighborhood_size)
            mean_sharpeness_image = np.round(np.mean(focus.mean(axis=(1, 2))),3)
            if mean_sharpeness_image > threshold:
                is_image_sharp = True
                list_sharp_images.append(image)
            else:
                is_image_sharp = False
            list_mean_sharpeness_image.append(mean_sharpeness_image)
            list_is_image_sharp.append(is_image_sharp)
        return list_mean_sharpeness_image, list_is_image_sharp,list_sharp_images
    
    def remove_outliers(self, array,min_percentile=1,max_percentile=98):
        max_val = np.percentile(array, max_percentile)
        if np.isnan(max_val) == True:
            max_val = np.percentile(array, max_percentile+0.1)
        min_val = np.percentile(array, min_percentile)
        if np.isnan(min_val) == True:
            min_val = np.percentile(array, min_percentile+0.1)
        array = array [array > min_val]
        array = array [array < max_val]
        return array 
    
    def is_None(self,variable_to_test):
        if (type(variable_to_test) is list):
            variable_to_test = variable_to_test[0]
        if variable_to_test in (None, 'None', 'none',['None'],['none'],[None]):
            is_none = True
        else:
            is_none = False
        return is_none
    
    def make_it_a_list(self,variable_to_test):
        if not (type(variable_to_test) is list):
            list_variable = [variable_to_test]
        else:
            list_variable = variable_to_test
        return list_variable
    
    # Function that reorder the index to make it continuos 
    def reorder_mask_image(self,mask_image_tested):
        number_masks = np.max(mask_image_tested)
        mask_new =np.zeros_like(mask_image_tested)
        if number_masks>0:
            counter = 0
            for index_mask in range(1,number_masks+1):
                if index_mask in mask_image_tested:
                    counter = counter + 1
                    if counter ==1:
                        mask_new = np.where(mask_image_tested == index_mask, -counter, mask_image_tested)
                    else:
                        mask_new = np.where(mask_new == index_mask, -counter, mask_new)
            reordered_mask = np.absolute(mask_new)
        else:
            reordered_mask = mask_new
        return reordered_mask  
    
    # Function that reorder the index to make it continuos 
    def remove_artifacts_from_mask_image(self,mask_image_tested, minimal_mask_area_size = 2000):
        number_masks = np.max(mask_image_tested)
        if number_masks>0:
            for index_mask in range(1,number_masks+1):
                mask_size = np.sum(mask_image_tested == index_mask)
                if mask_size <= minimal_mask_area_size:
                    #mask_image_tested = np.where(mask_image_tested == index_mask, mask_image_tested, 0)
                    mask_image_tested = np.where(mask_image_tested == index_mask,0,mask_image_tested )
            reordered_mask = Utilities().reorder_mask_image(mask_image_tested)
        else:
            reordered_mask=mask_image_tested
        return reordered_mask  
    
    def convert_to_standard_format(self,data_folder_path,path_to_config_file, number_color_channels=2,number_of_fov=1, download_data_from_NAS = True, use_metadata=False, is_format_FOV_Z_Y_X_C=True):
        path_to_masks_dir = None
        # Creating a folder to store all plots
        destination_folder = pathlib.Path().absolute().joinpath('temp_'+data_folder_path.name+'_sf')
        if pathlib.Path.exists(destination_folder):
            shutil.rmtree(str(destination_folder))
            destination_folder.mkdir(parents=True, exist_ok=True)
        else:
            destination_folder.mkdir(parents=True, exist_ok=True)
        
        local_data_dir, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
        if download_data_from_NAS == False:
            local_data_dir = data_folder_path
        # Downloading data
        if use_metadata == True:
            try:
                metadata = pycro.Dataset(str(local_data_dir))
                number_z_slices = max(metadata.axes['z'])+1
                number_color_channels = max(metadata.axes['channel'])+1
                number_of_fov = max(metadata.axes['position'])+1
                number_of_tp = max(metadata.axes['time'])+1
                detected_metadata = True
                print('Number of z slices: ', str(number_z_slices), '\n',
                    'Number of color channels: ', str(number_color_channels) , '\n'
                    'Number of FOV: ', str(number_of_fov) , '\n',
                    'Number of TimePoints', str(number_of_tp), '\n', '\n', '\n')
            except:
                raise ValueError('The metadata file is not found. Please check the path to the metadata file.')
        
        if is_format_FOV_Z_Y_X_C == True:
            #_, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
            number_images_all_fov = len(list_files_names_all_fov)
            # Re-arranging the image from shape [FOV, Z, Y, X, C] to multiple tifs with shape [Z, Y, X, C]  FOV_Z_Y_X_C
            list_images_standard_format= []
            list_files_names = []
            number_images =0
            for i in range(number_images_all_fov):
                for j in range (number_of_fov):
                    temp_image_fov = list_images_all_fov[i]
                    if use_metadata == False:
                        number_z_slices = temp_image_fov.shape[0]//2
                        if number_z_slices > 50:
                            raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    y_shape, x_shape = temp_image_fov.shape[2], temp_image_fov.shape[3]
                    list_files_names.append(  list_files_names_all_fov[i].split(".")[0]+'_fov_'+str(j) +'.tif' )
                    temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                    
                    temp_image = temp_image_fov[j,:,:,:] # format [Z,Y,X,C]
                    #for ch in range(number_color_channels):
                    #    temp_image[:,:,:,ch] = temp_image_fov[ch::number_color_channels,:,:] 
                    list_images_standard_format.append(temp_image)
                    number_images+=1
            
            for k in range(number_images):
                # image_name = list_files_names[i].split(".")[0] +'.tif'
                tifffile.imsave(str(destination_folder.joinpath(list_files_names[k])), list_images_standard_format[k])
            
            masks_dir = None
        else:
            if number_of_fov > 1:  
                # This option sections a single tif file containing multiple fov.
                # The format of the original FOV is [FOV_0:Ch_0-Ch_1-Z_1...Z_N, ... FOV_N:Ch_0-Ch_1-Z_1...Z_N]
                #local_data_dir, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
                #if download_data_from_NAS == False:
                #    local_data_dir = data_folder_path
                number_images_all_fov = len(list_files_names_all_fov)
                number_images = 0
                # This option sections a single tif file containing multiple fov.
                # The format of the original FOV is [FOV_0:Ch_0-Ch_1-Z_1...Z_N, ... FOV_N:Ch_0-Ch_1-Z_1...Z_N]  
                for k in range(number_images_all_fov):
                    # Section that separaters all fov into single tif files
                    image_with_all_fov = list_images_all_fov[k]
                    number_total_images_in_fov = image_with_all_fov.shape[0]
                    if detected_metadata == False:
                        if (number_total_images_in_fov % (number_color_channels*number_of_fov)) == 0:
                            number_z_slices = int(number_total_images_in_fov / (number_color_channels*number_of_fov))
                        else:
                            raise ValueError('The number of z slices is not defined correctly double-check the number_of_fov and number_color_channels.' )
                        if number_z_slices > 50:
                            raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    number_elements_on_fov = number_color_channels*number_z_slices
                    list_files_names = []
                    y_shape, x_shape = image_with_all_fov.shape[1], image_with_all_fov.shape[2]                
                    # Iterating for each image. Note that the color channels are intercalated in the original image. For that reason a for loop is needed and then selecting even and odd indexes.
                    list_images_standard_format= []
                    counter=0
                    for i in range(number_of_fov):
                        list_files_names.append(  list_files_names_all_fov[k].split(".")[0]+'_img_'+str(k)+'_fov_'+str(i) +'.tif' )
                        temp_image_fov = np.zeros((number_elements_on_fov,y_shape, x_shape))
                        temp_image_fov = image_with_all_fov[counter*number_elements_on_fov:number_elements_on_fov*(counter+1),:,:]
                        temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                        for ch in range(number_color_channels):
                            temp_image[:,:,:,ch] = temp_image_fov[ch::number_color_channels,:,:] 
                        #temp_image[:,:,:,0] = temp_image_fov[::2,:,:] # even indexes
                        #temp_image[:,:,:,1] = temp_image_fov[1::2,:,:] # odd indexes
                        list_images_standard_format.append(temp_image)
                        counter+=1
                        number_images+=1
                        del temp_image, temp_image_fov
            elif number_of_fov == 1:
                # This option takes multiple tif files containing multiple images with format [FOV_0:Ch_0-Ch_1-Z_1...Z_N, ... FOV_N:Ch_0-Ch_1-Z_1...Z_N]
                #_, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
                number_images = len(list_files_names_all_fov)
                # Re-arranging the image
                list_images_standard_format= []
                list_files_names = []
                for i in range(number_images):
                    temp_image_fov = list_images_all_fov[i]
                    number_z_slices = temp_image_fov.shape[0]//2
                    if number_z_slices > 50:
                        raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    y_shape, x_shape = temp_image_fov.shape[1], temp_image_fov.shape[2]
                    list_files_names.append(  list_files_names_all_fov[i].split(".")[0]+'_fov_'+str(i) +'.tif' )
                    temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                    for ch in range(number_color_channels):
                        temp_image[:,:,:,ch] = temp_image_fov[ch::number_color_channels,:,:] 
                    #temp_image[:,:,:,0] = list_images_all_fov[i][::2,:,:] # even indexes
                    #temp_image[:,:,:,1] = list_images_all_fov[i][1::2,:,:] # odd indexes
                    list_images_standard_format.append(temp_image)
            
        
        # Saving images as tif files
        for i in range(number_images):
            # image_name = list_files_names[i].split(".")[0] +'.tif'
            tifffile.imsave(str(destination_folder.joinpath(list_files_names[i])), list_images_standard_format[i])
        masks_dir = None
        return destination_folder,masks_dir, list_files_names, list_images_all_fov, list_images_standard_format, number_of_fov, number_color_channels, number_z_slices, number_of_tp
    
    def create_output_folders(self,data_folder_path,diameter_nucleus,diameter_cytosol,psf_z,psf_yx,threshold_for_spot_detection,channels_with_FISH,list_threshold_for_spot_detection):
        # testing if the images were merged.
        if data_folder_path.name == 'merged':
            data_folder_path = data_folder_path.parents[0]
        # Testing if a temporal folder was created.
        if data_folder_path.name[0:5] == 'temp_':
            original_folder_name = data_folder_path.name[5:]
        else:
            original_folder_name= data_folder_path.name
        # Creating the output_identification_string
        if (threshold_for_spot_detection is None):
            output_identification_string = original_folder_name+'___nuc_' + str(diameter_nucleus) +'__cyto_' + str(diameter_cytosol) +'__psfz_' + str(psf_z) +'__psfyx_' + str(psf_yx)+'__ts_auto'
        else:
            output_identification_string = original_folder_name +'___nuc_' + str(diameter_nucleus) +'__cyto_' + str(diameter_cytosol) +'__psfz_' + str(psf_z) +'__psfyx_' + str(psf_yx)+'__ts'
            for i in range (len(channels_with_FISH)):
                output_identification_string+='_'+ str(list_threshold_for_spot_detection[i])
                print ('\n Output folder name : ' , output_identification_string)
        # Output folders
        analysis_folder_name = 'analysis_'+ output_identification_string
        # Removing directory if exist
        if os.path.exists(analysis_folder_name):
            shutil.rmtree(analysis_folder_name)
        # Creating the directory
        os.makedirs(analysis_folder_name) 
        return output_identification_string
    
    
    def  create_list_thresholds_FISH(self,channels_with_FISH,threshold_for_spot_detection=None):
        # If more than one channel contain FISH spots. This section will create a list of thresholds for spot detection and for each channel. 
        if not(isinstance(channels_with_FISH, list)):
            channels_with_FISH=Utilities().make_it_a_list(channels_with_FISH)
        list_threshold_for_spot_detection=[]
        if not isinstance(threshold_for_spot_detection, list):
            for i in range (len(channels_with_FISH)):
                list_threshold_for_spot_detection.append(threshold_for_spot_detection)
        else:
            list_threshold_for_spot_detection = threshold_for_spot_detection
        # Lists for thresholds. If the list is smaller than the number of FISH channels and it uses the same value for all channels.
        if (isinstance(list_threshold_for_spot_detection, list)) and (len(list_threshold_for_spot_detection) < len(channels_with_FISH)):
            for i in range (len(channels_with_FISH)):
                list_threshold_for_spot_detection.append(list_threshold_for_spot_detection[0])
        return list_threshold_for_spot_detection
    
    # This function is intended to merge masks in a single image
    def merge_masks (self,list_masks):
        '''
        This method is intended to merge a list of images into a single image (Numpy array) where each cell is represented by an integer value.
        
        Parameters
        
        list_masks : List of Numpy arrays.
            List of Numpy arrays, where each array has dimensions [Y, X] with values 0 and 1, where 0 represents the background and 1 the cell mask in the image.
        '''
        n_masks = len(list_masks)
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
            im_zeros = np.zeros_like(image)
            for ch in range( image.shape[2]):
                if np.max(image[:,:,ch]) >0:
                    im_zeros[:,:,ch] = RemoveExtrema(image[:,:,ch],min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers() 
            image = im_zeros
        image_new= np.zeros_like(image)
        for i in range(0, image.shape[2]):  # iterate for each channel
            if np.max(image[:,:,i]) >0:
                temp = image[:,:,i].copy()
                image_new[:,:,i]= ( (temp-np.min(temp))/(np.max(temp)-np.min(temp)) ) * 255
            image_new = np.uint8(image_new)
        # padding with zeros the channel dimension.
        while image_new.shape[2]<3:
            zeros_plane = np.zeros_like(image_new[:,:,0])
            image_new = np.concatenate((image_new,zeros_plane[:,:,np.newaxis]),axis=2)
        return image_new

    def read_zipfiles_from_NAS(self,list_dirs,path_to_config_file,share_name,mandatory_substring,local_folder_path):
        # This function iterates over all zip files in a remote directory and download them to a local directory
        list_remote_files=[]
        list_local_files =[]
        if (isinstance(list_dirs, tuple)==False) and (isinstance(list_dirs, list)==False):
            list_dirs = [list_dirs]
        for folder in list_dirs:
            print(folder)
            list_files = NASConnection(path_to_config_file,share_name = share_name).read_files(folder,timeout=60)
            for file in list_files:
                if ('.zip' in file) and (mandatory_substring in file):   # add an argument with re conditions 
                    # Listing all zip files
                    zip_file_path = pathlib.Path().joinpath(folder,file)
                    list_remote_files.append (zip_file_path)
                    list_local_files.append(pathlib.Path().joinpath(local_folder_path,zip_file_path.name)) 
                    # downloading the zip files from NAS
                    NASConnection(path_to_config_file,share_name = share_name).download_file(zip_file_path, local_folder_path,timeout=200)
        return list_local_files
    
    def unzip_local_folders(self,list_local_files,local_folder_path):
        list_local_folders =[]
        for zip_folder in list_local_files:
            # Reads from a list of zip files
            file_to_unzip = zipfile.ZipFile(str(zip_folder)) # opens zip
            temp_folder_name = pathlib.Path().joinpath(local_folder_path, zip_folder.stem)
            if (os.path.exists(temp_folder_name)) :
                shutil.rmtree(temp_folder_name)
                os.makedirs(temp_folder_name) # make a new directory
            # Iterates for each file in zip file
            for file_in_zip in file_to_unzip.namelist():
                # Extracts data to specific folder
                file_to_unzip.extract(file_in_zip,temp_folder_name)
            # Closes the zip file
            file_to_unzip.close()
            # removes the original zip file
            os.remove(pathlib.Path().joinpath(local_folder_path, zip_folder.name))
            list_local_folders.append(temp_folder_name)
        return list_local_folders
    
    def dataframe_extract_data(self,dataframe,spot_type, minimum_spots_cluster=2):
        ''' This function is intended to read a dataframe and returns 
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size
        '''
        # Number of cells
        number_cells = dataframe['cell_id'].nunique()
        # Number of spots in cytosol
        number_of_spots_per_cell_cytosol = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i) & (dataframe['is_nuc']==False) & (dataframe['spot_type']==spot_type)  & (dataframe['is_cell_fragmented']!=-1) ].spot_id) for i in range(0, number_cells)])
        # Number of spots in nucleus.  Spots without TS.
        number_of_spots_per_cell_nucleus = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i) & (dataframe['is_nuc']==True) & (dataframe['spot_type']==spot_type)  & (dataframe['is_cell_fragmented']!=-1)    ].spot_id) for i in range(0, number_cells)])
        # Number of spots
        number_of_spots_per_cell = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i)  & (dataframe['spot_type']==spot_type) & (dataframe['is_cell_fragmented']!=-1)].spot_id) for i in range(0, number_cells)])
        # Number of TS per cell.
        number_of_TS_per_cell = [len( dataframe.loc[  (dataframe['cell_id']==i) &  (dataframe['is_cluster']==True) & (dataframe['is_nuc']==True) & (dataframe['spot_type']==spot_type)  &   (dataframe['cluster_size']>=minimum_spots_cluster)  & (dataframe['is_cell_fragmented']!=-1)  ].spot_id) for i in range(0, number_cells)]
        number_of_TS_per_cell= np.asarray(number_of_TS_per_cell)
        # Number of RNA in a TS
        ts_size =  dataframe.loc[ (dataframe['is_cluster']==True) & (dataframe['is_nuc']==True)  & (dataframe['spot_type']==spot_type) &   (dataframe['cluster_size']>=minimum_spots_cluster)  & (dataframe['is_cell_fragmented']!=-1)   ].cluster_size.values
        # Size of each cell
        cell_size = [dataframe.loc[   (dataframe['cell_id']==i) ].cell_area_px.values[0] for i in range(0, number_cells)]
        cell_size = np.asarray(cell_size)
        # Cyto size
        cyto_size = [dataframe.loc[   (dataframe['cell_id']==i) ].cyto_area_px.values[0] for i in range(0, number_cells)]
        cyto_size = np.asarray(cyto_size)
        # Size of the nucleus of each cell
        nuc_size = [dataframe.loc[   (dataframe['cell_id']==i) ].nuc_area_px.values[0] for i in range(0, number_cells)]
        nuc_size = np.asarray(nuc_size)
        # removing values less than zeros
        number_of_spots_per_cell.clip(0)
        number_of_spots_per_cell_cytosol.clip(0)
        number_of_spots_per_cell_nucleus.clip(0)
        number_of_TS_per_cell.clip(0)
        ts_size.clip(0)
        return number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size,cell_size, number_cells, nuc_size, cyto_size
    
    def extracting_data_for_each_df_in_directory(self,list_local_folders, current_dir,spot_type=0, minimum_spots_cluster=2):
        '''
        This method is intended to extract data from the dataframe
        '''
        # Extracting data from dataframe and converting it into lists for each directory.
        list_spots_total=[]
        list_spots_nuc=[]
        list_spots_cytosol=[]
        list_number_cells =[]
        list_transcription_sites =[]
        list_cell_size=[]
        list_nuc_size =[]
        list_dataframes =[]
        list_cyto_size =[]
        for i in range (0, len (list_local_folders)):
            dataframe_dir = current_dir.joinpath('analyses',list_local_folders[i])    # loading files from "analyses" folder
            dataframe_file = glob.glob( str(dataframe_dir.joinpath('dataframe_*')) )[0]
            dataframe = pd.read_csv(dataframe_file)
            # Extracting values from dataframe
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells, nuc_size, cyto_size = Utilities().dataframe_extract_data(dataframe,spot_type,minimum_spots_cluster=minimum_spots_cluster)            
            # Appending each condition to a list
            list_spots_total.append(number_of_spots_per_cell)  # This list includes spots and TS in the nucleus
            list_spots_nuc.append(number_of_spots_per_cell_nucleus)   #
            list_spots_cytosol.append(number_of_spots_per_cell_cytosol)
            list_number_cells.append(number_cells)
            list_transcription_sites.append(number_of_TS_per_cell)
            list_cell_size.append(cell_size)
            list_nuc_size.append(nuc_size)
            list_dataframes.append(dataframe)
            list_cyto_size.append(cyto_size)
            # Deleting variables
            del number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells,nuc_size
        return list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites,list_cell_size,list_dataframes,list_nuc_size,list_cyto_size
    
    def extract_data_interpretation(self,list_dirs, path_to_config_file, current_dir, mandatory_substring, local_folder_path, list_labels, share_name='share',minimum_spots_cluster=2, connect_to_NAS=0, spot_type=0, remove_extreme_values=False):
        if connect_to_NAS == True:
            # Reading the data from NAS, unziping files, organizing data as single dataframe for comparison. 
            list_local_files = Utilities().read_zipfiles_from_NAS(list_dirs,path_to_config_file,share_name, mandatory_substring, local_folder_path)
            list_local_folders = Utilities().unzip_local_folders(list_local_files,local_folder_path)
        else: 
            list_local_folders = list_dirs # Use this line to process files from a local repository
        # Extracting data from each repository
        list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites, list_cell_size, list_dataframes, list_nuc_size, list_cyto_size = Utilities().extracting_data_for_each_df_in_directory(  list_local_folders=list_local_folders,current_dir=current_dir,spot_type=spot_type,minimum_spots_cluster=minimum_spots_cluster)
        # Final dataframes for nuc, cyto and total spots
        df_all = Utilities().convert_list_to_df (list_number_cells, list_spots_total, list_labels, remove_extreme_values= remove_extreme_values)
        df_cyto = Utilities().convert_list_to_df (list_number_cells, list_spots_cytosol, list_labels, remove_extreme_values= remove_extreme_values)
        df_nuc = Utilities().convert_list_to_df (list_number_cells, list_spots_nuc, list_labels, remove_extreme_values= remove_extreme_values)
        df_transcription_sites = Utilities().convert_list_to_df (list_number_cells, list_transcription_sites, list_labels, remove_extreme_values= remove_extreme_values)
        return df_all, df_cyto, df_nuc, df_transcription_sites, list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites, list_cell_size, list_dataframes, list_nuc_size, list_cyto_size 
    
    def function_get_df_columns_as_array(self,df, colum_to_extract, extraction_type='all_values'):
        '''This method is intended to extract a column from a dataframe and convert its values to an array format.
            The argument <<<extraction_type>>> accepts two possible values. 
                values_per_cell: this returns an unique value that represents a cell parameter and is intended to be used with the following columns 
                        'nuc_int_ch", cyto_int_ch', 'nuc_loc_y', 'nuc_loc_x', 'cyto_loc_y', 'cyto_loc_x', 'nuc_area_px', 'cyto_area_px', 'cell_area_px'
                all_values: this returns all fields in the dataframe for the specified column.  
        '''
        number_cells = df['cell_id'].nunique()
        if extraction_type == 'values_per_cell':
            return np.asarray( [       df.loc[(df['cell_id']==i)][colum_to_extract].values[0]            for i in range(0, number_cells)] )
        elif extraction_type == 'all_values' :
            return np.asarray( [       df.loc[(df['cell_id']==i)][colum_to_extract].values          for i in range(0, number_cells)] )      
    
    def convert_list_to_df (self,list_number_cells, list_spots, list_labels, remove_extreme_values= False,max_quantile=0.98) :
        # defining the dimensions for the array.
        max_number_cells = max(list_number_cells)
        number_conditions = len(list_number_cells)
        # creating an array with the same dimensions
        spots_array = np.empty((max_number_cells, number_conditions))
        spots_array[:] = np.NaN
        # replace the elements in the array
        for i in range(0, number_conditions ):
            spots_array[0:list_number_cells[i], i] = list_spots[i] 
        # creating a dataframe
        df = pd.DataFrame(data = spots_array, columns=list_labels)
        # Removing 1% extreme values.
        if remove_extreme_values == True:
            for col in df.columns:
                max_data_value = df[col].quantile(max_quantile)
                df[col] = np.where(df[col] >= max_data_value, np.nan, df[col])
        return df

    def download_data_NAS(self,path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200):
        '''
        This method is inteded to download data from a NAS. to a local directory.
        path_to_config_file
        data_folder_path
        path_to_masks_dir
        share_name,timeout
        '''
        # Downloading data from NAS
        local_folder_path = pathlib.Path().absolute().joinpath('temp_' + data_folder_path.name)
        NASConnection(path_to_config_file,share_name = share_name).copy_files(data_folder_path, local_folder_path,timeout=timeout)
        local_data_dir = local_folder_path     # path to a folder with images.
        # Downloading masks from NAS
        if not (path_to_masks_dir is None):
            local_folder_path_masks = pathlib.Path().absolute().joinpath( path_to_masks_dir.stem  )
            zip_file_path = local_folder_path_masks.joinpath( path_to_masks_dir.stem +'.zip')
            NASConnection(path_to_config_file,share_name = share_name).download_file(path_to_masks_dir, local_folder_path_masks,timeout=timeout)
            # Unzip downloaded images and update mask directory
            file_to_unzip = zipfile.ZipFile(str(zip_file_path)) # opens zip
            # Iterates for each file in zip file
            for file_in_zip in file_to_unzip.namelist():
                # Extracts data to specific folder
                file_to_unzip.extract(file_in_zip,local_folder_path_masks)
            # Closes the zip file
            file_to_unzip.close()
            # removes the original zip file
            os.remove(zip_file_path)
            masks_dir = local_folder_path_masks
        else:
            masks_dir = None
        return local_data_dir, masks_dir
    
    def read_images_from_folder(self, path_to_config_file, data_folder_path, path_to_masks_dir=None, download_data_from_NAS=False, substring_to_detect_in_file_name = '.*_C0.tif'):
        # Download data from NAS
        if download_data_from_NAS == True:
            share_name = 'share'
            local_data_dir, masks_dir = Utilities().download_data_NAS(path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200)
        else:
            local_data_dir = data_folder_path 
            masks_dir = path_to_masks_dir 
        # Detecting if images need to be merged
        is_needed_to_merge_images = MergeChannels(local_data_dir, substring_to_detect_in_file_name = substring_to_detect_in_file_name, save_figure =1).checking_images()
        if is_needed_to_merge_images == True:
            _, _, number_images, _ = MergeChannels(local_data_dir, substring_to_detect_in_file_name = substring_to_detect_in_file_name, save_figure =1).merge()
            local_data_dir = local_data_dir.joinpath('merged')
            list_images, path_files, list_files_names, number_images = ReadImages(directory= local_data_dir).read()
        else:
            list_images, path_files, list_files_names, number_images = ReadImages(directory= local_data_dir).read()  # list_images, path_files, list_files_names, number_files        
        # Printing image properties
        if len(list_images[0].shape) < 4:
            number_color_channels = None
        else:
            number_color_channels = list_images[0].shape[-1] 
        print('Image shape: ', list_images[0].shape , '\n')
        print('Number of images: ',number_images , '\n')
        print('Local directory with images: ', local_data_dir, '\n')
        return local_data_dir, masks_dir, number_images, number_color_channels, list_files_names,list_images
        
    def save_output_to_folder (self, output_identification_string, data_folder_path,
                                list_files_distributions=None,
                                file_plots_bleed_thru = None,
                                file_plots_int_ratio=None,
                                file_plots_int_pseudo_ratio=None,
                                channels_with_FISH=None,
                                save_pdf_report=True):
        #  Moving figures to the final folder 
        if not (list_files_distributions is None) and (type(list_files_distributions) is list):
            file_plots_distributions = list_files_distributions[0]
            file_plots_cell_size_vs_num_spots = list_files_distributions[1]
            file_plots_cell_intensity_vs_num_spots = list_files_distributions[2]
            file_plots_spot_intensity_distributions = list_files_distributions[3]
            for i in range (len(file_plots_distributions)):
                if not (file_plots_distributions is None):
                    pathlib.Path().absolute().joinpath(file_plots_distributions[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_distributions[i]))
                if not (file_plots_cell_size_vs_num_spots is None):
                    pathlib.Path().absolute().joinpath(file_plots_cell_size_vs_num_spots[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_cell_size_vs_num_spots[i]))
                if not (file_plots_cell_intensity_vs_num_spots is None):
                    pathlib.Path().absolute().joinpath(file_plots_cell_intensity_vs_num_spots[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_cell_intensity_vs_num_spots[i]))
                if not (file_plots_spot_intensity_distributions is None):
                    pathlib.Path().absolute().joinpath(file_plots_spot_intensity_distributions[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_spot_intensity_distributions[i]))
            
        if not (file_plots_bleed_thru is None):
            pathlib.Path().absolute().joinpath(file_plots_bleed_thru).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_bleed_thru))
        if not (file_plots_int_ratio is None):
            pathlib.Path().absolute().joinpath(file_plots_int_ratio).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_int_ratio))
        if not (file_plots_int_pseudo_ratio is None):
            pathlib.Path().absolute().joinpath(file_plots_int_pseudo_ratio).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_int_pseudo_ratio))
        
        # all original images
        pathlib.Path().absolute().joinpath('original_images_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'original_images_'+ data_folder_path.name +'.pdf'))
        # all cell images
        for i in range (len(channels_with_FISH)):
            temp_plot_name = 'cells_channel_'+ str(channels_with_FISH[i])+'_'+ data_folder_path.name +'.pdf'
            pathlib.Path().absolute().joinpath(temp_plot_name).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),temp_plot_name))
        #metadata_path
        pathlib.Path().absolute().joinpath('images_report_'+ data_folder_path.name +'.csv').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'images_report_'+ data_folder_path.name +'.csv'))
        pathlib.Path().absolute().joinpath('metadata_'+ data_folder_path.name +'.txt').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'metadata_'+ data_folder_path.name +'.txt'))
        #dataframe_path 
        pathlib.Path().absolute().joinpath('dataframe_' + data_folder_path.name +'.csv').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'dataframe_'+ data_folder_path.name +'.csv'))
        #pdf_path 
        if save_pdf_report == True:
            pathlib.Path().absolute().joinpath('pdf_report_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'pdf_report_'+ data_folder_path.name +'.pdf'))
        #pdf_path segmentation 
        pathlib.Path().absolute().joinpath('segmentation_images_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'segmentation_images_'+ data_folder_path.name +'.pdf'))

        return None

    def sending_data_to_NAS(self,output_identification_string, data_folder_path, path_to_config_file, path_to_masks_dir, diameter_nucleus, diameter_cytosol, send_data_to_NAS = False, masks_dir = None, share_name = 'share'):
        # Writing analyses data to NAS
        analysis_folder_name = 'analysis_'+ output_identification_string
        if send_data_to_NAS == True:
            shutil.make_archive(analysis_folder_name,'zip',pathlib.Path().absolute().joinpath(analysis_folder_name))
            local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip')
            NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
            os.remove(pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip'))
        # Writing masks to NAS
        ## Creating mask directory name
        if path_to_masks_dir == None: 
            mask_folder_created_by_pipeline = 'masks_'+ data_folder_path.name # default name by pipeline
            name_final_masks = data_folder_path.name +'___nuc_' + str(diameter_nucleus) + '__cyto_' + str(diameter_cytosol) 
            mask_dir_complete_name = 'masks_'+ name_final_masks # final name for masks dir
            shutil.move(mask_folder_created_by_pipeline, mask_dir_complete_name ) # remaing the masks dir
        elif masks_dir is None:
            mask_dir_complete_name = None
        else: 
            mask_dir_complete_name = masks_dir.name
        ## Sending masks to NAS
        if (send_data_to_NAS == True) and (path_to_masks_dir == None) :
            shutil.make_archive( mask_dir_complete_name , 'zip', pathlib.Path().absolute().joinpath(mask_dir_complete_name))
            local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip')
            NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
            os.remove(pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip'))
        return analysis_folder_name, mask_dir_complete_name
    
    def move_results_to_analyses_folder(self, output_identification_string,  data_folder_path,mask_dir_complete_name,path_to_masks_dir, save_filtered_images = False, download_data_from_NAS = False, save_masks_as_file = False):
        # Moving all results to "analyses" folder
        if not os.path.exists(str('analyses')):
            os.makedirs(str('analyses'))
        # Subfolder name
        analysis_folder_name = 'analysis_'+ output_identification_string
        final_dir_name =pathlib.Path().absolute().joinpath('analyses', analysis_folder_name)
        # Removing directory if exist
        if os.path.exists(str(final_dir_name)):
            shutil.rmtree(str(final_dir_name))
        # Moving results to a subdirectory in 'analyses' folder
        pathlib.Path().absolute().joinpath(analysis_folder_name).rename(final_dir_name )
        # Moving masks to a subdirectory in 'analyses' folder
        if (download_data_from_NAS == True or path_to_masks_dir is None) and save_masks_as_file:
            final_mask_dir_name = pathlib.Path().absolute().joinpath('analyses', mask_dir_complete_name)
            if os.path.exists(str(final_mask_dir_name)):
                shutil.rmtree(str(final_mask_dir_name))
            pathlib.Path().absolute().joinpath(mask_dir_complete_name).rename(final_mask_dir_name )
        if save_filtered_images == True:
            filtered_folder_name = 'filtered_images_' + data_folder_path.name 
            pathlib.Path().absolute().joinpath(filtered_folder_name).rename(pathlib.Path().absolute().joinpath('analyses',str('analysis_'+ output_identification_string    ),filtered_folder_name))
        # Delete local temporal files
        temp_results_folder_name = pathlib.Path().absolute().joinpath('temp_results_' + data_folder_path.name)
        shutil.rmtree(temp_results_folder_name)
        # Removing local folder
        # Removing directory if exist
        std_format_folder_name = 'temp_'+data_folder_path.name+'_sf'
        std_format_folder_name_dir_name =pathlib.Path().absolute().joinpath(std_format_folder_name)
        if os.path.exists(str(std_format_folder_name_dir_name)):
            shutil.rmtree(str(std_format_folder_name_dir_name))
        
        if (download_data_from_NAS == True):
            # Delete temporal images downloaded from NAS
            shutil.rmtree('temp_'+data_folder_path.name)
        return None
    
    def export_data_to_CSV(self,list_spots_total, list_spots_nuc, list_spots_cytosol, destination_folder, plot_title_suffix=''):
        # Exporting data to CSV. 
        # ColumnA = time, 
        # ColumnB= #RNA in nucleus, 
        # ColumnC = #RNA in cytoplasm, 
        # ColumnD = total RNA.
        num_time_points = len(list_spots_total)
        num_columns = 4 # time, RNA_nuc, RNA_cyto, total
        array_data_spots =  np.empty(shape=(0, num_columns))
        for i in range(0, num_time_points):
            num_cells = len(list_spots_total[i])
            temp_array_data_spots = np.zeros((num_cells,num_columns))
            temp_array_data_spots[:,0] = i
            temp_array_data_spots[:,1] = list_spots_nuc[i] # nuc
            temp_array_data_spots[:,2] = list_spots_cytosol[i] # cyto
            temp_array_data_spots[:,3] = list_spots_total[i] # all spots
            array_data_spots = np.append(array_data_spots, temp_array_data_spots, axis=0)
        array_data_spots.shape
        # final data frame with format for the model
        df_for_model = pd.DataFrame(data=array_data_spots, columns =['time_index', 'RNA_nuc','RNA_cyto','RNA_total'] )
        new_dtypes = {'time_index':int, 'RNA_nuc':int, 'RNA_cyto':int,'RNA_total':int}
        df_for_model = df_for_model.astype(new_dtypes)
        # Save to csv
        df_for_model.to_csv(pathlib.Path().absolute().joinpath(destination_folder,plot_title_suffix+'.csv'))
        return df_for_model
    def extract_images_masks_dataframe( self,data_folder_path, mandatory_substring, path_to_config_file,connect_to_NAS,path_to_masks_dir=None, rescale=False,max_percentile=99.5):
        local_folder_path = pathlib.Path().absolute().joinpath('temp_local__'+data_folder_path.name)
        # This section downloads results including the dataframe
        if connect_to_NAS == True:
            list_local_files = Utilities().read_zipfiles_from_NAS(list_dirs=data_folder_path,path_to_config_file=path_to_config_file,share_name='share', mandatory_substring=mandatory_substring, local_folder_path=local_folder_path)
            list_local_folders = Utilities().unzip_local_folders(list_local_files,local_folder_path)
        else: 
            list_local_folders = data_folder_path # Use this line to process files from a local repository
        # Extracting the dataframe
        dataframe_file_path = glob.glob( str(list_local_folders[0].joinpath('dataframe_*')) )[0]
        dataframe = pd.read_csv(dataframe_file_path)
        # Extracting the dataframe with cell ids
        try:
            dataframe_file_path_metadata = glob.glob( str(list_local_folders[0].joinpath('images_report_*')) )[0]     
            images_metadata = pd.read_csv(dataframe_file_path_metadata)
        except:
            images_metadata = None
        # Extracting Original images
        local_data_dir, masks_dir, number_images, number_color_channels, list_files_names,_ = Utilities().read_images_from_folder( path_to_config_file, data_folder_path = data_folder_path, path_to_masks_dir = path_to_masks_dir,  download_data_from_NAS = connect_to_NAS, substring_to_detect_in_file_name = '.*_C0.tif')        
        # Reading images from folders
        list_images, path_files, list_files_names, _ = ReadImages(directory= local_data_dir).read()
        if not (path_to_masks_dir is None):
            list_masks, path_files_masks, list_files_names_masks, _ = ReadImages(directory= masks_dir).read()
        else:
            list_masks = None
        # Converting the images to int8
        return list_images, list_masks, dataframe, number_images, number_color_channels,list_local_folders,local_data_dir, images_metadata
    
    
    def image_cell_selection(self,cell_id, list_images, dataframe, mask_cell=None, mask_nuc=None, scaling_value_radius_cell=1.1):
        SCALING_RADIUS_NUCLEUS = scaling_value_radius_cell #1.1
        SCALING_RADIUS_CYTOSOL = scaling_value_radius_cell
        # selecting only the dataframe containing the values for the selected field
        df_selected_cell = dataframe.loc[   (dataframe['cell_id']==cell_id)]
        selected_image_id = df_selected_cell.image_id.values[0]
        y_max_image_shape = list_images[selected_image_id].shape[1]-1
        x_max_image_shape = list_images[selected_image_id].shape[2]-1
        # Cell location in image
        scaling_value_radius_cell = scaling_value_radius_cell # use this parameter to increase or decrease the number of radius to plot from the center of the cell.
        nuc_loc_x = df_selected_cell.nuc_loc_x.values[0]
        nuc_loc_y = df_selected_cell.nuc_loc_y.values[0]
        cyto_loc_x = df_selected_cell.cyto_loc_x.values[0]
        cyto_loc_y = df_selected_cell.cyto_loc_y.values[0]
        nuc_radius_px =  int(np.sqrt(df_selected_cell.nuc_area_px.values[0])*SCALING_RADIUS_NUCLEUS)
        cyto_radius_px = int(np.sqrt(df_selected_cell.cyto_area_px.values[0])*SCALING_RADIUS_CYTOSOL)
        # Detecting if a mask for the cytosol was used. If true, the code will plot the complete cell. Else, it will only plot the cell nucleus.
        if cyto_loc_x:
            plot_complete_cell = True
        else:
            plot_complete_cell = False
        if plot_complete_cell == True:
            x_min_value = cyto_loc_x - cyto_radius_px
            x_max_value = cyto_loc_x + cyto_radius_px
            y_min_value = cyto_loc_y - cyto_radius_px
            y_max_value = cyto_loc_y + cyto_radius_px
        else:
            x_min_value = nuc_loc_x - nuc_radius_px
            x_max_value = nuc_loc_x + nuc_radius_px
            y_min_value = nuc_loc_y - nuc_radius_px
            y_max_value = nuc_loc_y + nuc_radius_px
        # making sure that the selection doesnt go outside the limits of the original image
        x_min_value = np.max((0,x_min_value ))
        y_min_value = np.max((0,y_min_value ))
        x_max_value = np.min((x_max_value,x_max_image_shape))
        y_max_value = np.min((y_max_value,y_max_image_shape))
        # coordinates to select in the image 
        subsection_image_with_selected_cell = list_images[selected_image_id][:,y_min_value: y_max_value,x_min_value:x_max_value,:]
        # coordinates to select in the masks image
        if not (mask_cell is None):
            subsection_mask_cell = mask_cell[y_min_value: y_max_value,x_min_value:x_max_value]
            subsection_mask_cell[0, :] = 0; subsection_mask_cell[-1, :] = 0; subsection_mask_cell[:, 0] = 0; subsection_mask_cell[:, -1] = 0
        else:
            subsection_mask_cell = None
        if not (mask_nuc is None):
            subsection_mask_nuc = mask_nuc[y_min_value: y_max_value,x_min_value:x_max_value]
            subsection_mask_nuc[0, :] = 0; subsection_mask_nuc[-1, :] = 0; subsection_mask_nuc[:, 0] = 0; subsection_mask_nuc[:, -1] = 0
        else:
            subsection_mask_nuc = None 
        # spots
        df_spots = df_selected_cell[['spot_id', 'z', 'y', 'x','is_nuc', 'is_cluster','cluster_size','spot_type']]
        df_spots = df_spots.reset_index(drop=True)
        # Removing columns with -1. 
        df_spots = df_spots[df_spots.spot_id >= 0]
        # Re-organizing the origin of the image based on the subsection.
        df_spots_subsection_coordinates = df_spots.copy()
        df_spots_subsection_coordinates['y'] = df_spots_subsection_coordinates['y'] - y_min_value
        df_spots_subsection_coordinates['x'] = df_spots_subsection_coordinates['x'] - x_min_value
        return subsection_image_with_selected_cell, df_spots_subsection_coordinates,subsection_mask_cell, subsection_mask_nuc,selected_image_id
    
    
    def extract_spot_location_from_cell(self,df, spot_type=0, min_ts_size= None,z_slice=None):
        df_spots_all_z = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) ] 
        number_spots = len (  df_spots_all_z  )
        
        # Locating spots in the dataframe
        if (z_slice is None):
            df_spots = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) ] 
        else:
            df_spots = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) & (df['z']==z_slice) ] 
        number_spots_selected_z = len (  df_spots  )
        
        if number_spots_selected_z >0:
            y_spot_locations = df_spots['y'].values
            x_spot_locations = df_spots['x'].values
        else:
            y_spot_locations = None
            x_spot_locations = None
        
        # locating the TS in  the dataframe 
        if not (min_ts_size is None):
            df_TS = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==1) & (df['cluster_size']>=min_ts_size) ] 
        else:
            df_TS = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==1)]
        number_TS = len (  df_TS  )
        # TS location
        if number_TS >0:
            y_TS_locations = df_TS['y'].values
            x_TS_locations = df_TS['x'].values
        else:
            y_TS_locations = None
            x_TS_locations = None
        return y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations,number_spots, number_TS,number_spots_selected_z
    
    def spot_crops (self,image,df,number_crops_to_show,spot_size=5):
        number_crops_to_show = np.min((number_crops_to_show, len(df)/2))
        def return_crop(image, x, y,spot_size):
            spot_range = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
            crop_image = image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()
            return crop_image
        list_crops_0 = []
        list_crops_1 = []
        counter =0
        i =0
        while counter <= number_crops_to_show:
            x_co = df['x'].values[i]
            y_co = df['y'].values[i]
            if (x_co> (spot_size - 1) / 2) and (y_co>(spot_size - 1) / 2):
                crop_0 = return_crop(image=image[:,:,0], x=x_co, y=y_co, spot_size=spot_size)
                crop_1 = return_crop(image=image[:,:,1], x=x_co, y=y_co, spot_size=spot_size)
                list_crops_0.append(crop_0) 
                list_crops_1.append(crop_1)
                counter+=1
            i+=1
        return list_crops_0, list_crops_1
