import datetime
import getpass
import os
import platform
import socket
import sys

import numpy as np
import pandas as pd
import pkg_resources

from src.Util.ReadImages import ReadImages


# from ReadImages import ReadImages


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
    list_counter_cell_id : str
        Counter that keeps track of the number of images in the folder.
    threshold_for_spot_detection : int
        Threshold value used to discriminate background noise from mRNA spots in the image.
    '''
    def __init__(self,data_dir, channels_with_cytosol, channels_with_nucleus, channels_with_FISH, diameter_nucleus, diameter_cytosol, minimum_spots_cluster, list_voxels=None, list_psfs=None, file_name_str=None,list_segmentation_successful=True,list_counter_image_id=[],threshold_for_spot_detection=[],number_of_images_to_process=None,remove_z_slices_borders=False,NUMBER_Z_SLICES_TO_TRIM=0,CLUSTER_RADIUS=0,list_thresholds_spot_detection=[None],list_average_spots_per_cell=[None],list_number_detected_cells=[None],list_is_image_sharp=[None],list_metric_sharpeness_images=[None],remove_out_of_focus_images=False,sharpness_threshold=None):
        
        self.list_images, self.path_files, self.list_files_names, self.number_images = ReadImages(data_dir,number_of_images_to_process).read()
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
        self.threshold_for_spot_detection=threshold_for_spot_detection
        if  (not str(data_dir.name)[0:5] ==  'temp_') and (self.file_name_str is None):
            self.filename = 'metadata_'+ str(data_dir.name).replace(" ", "")  +'.txt'
            self.filename_csv = 'images_report_'+ str(data_dir.name).replace(" ", "")+'.csv'
        elif not(self.file_name_str is None):
            self.filename = 'metadata_'+ str(file_name_str).replace(" ", "") +'.txt'
            self.filename_csv = 'images_report_'+ str(file_name_str).replace(" ", "")+'.csv'
        else:
            self.filename = 'metadata_'+ str(data_dir.name[5:].replace(" ", "")) +'.txt'
            self.filename_csv = 'images_report_'+ str(file_name_str).replace(" ", "")+'.csv'
        self.data_dir = data_dir
        self.list_segmentation_successful =list_segmentation_successful
        self.list_counter_image_id=list_counter_image_id
        self.remove_z_slices_borders=remove_z_slices_borders
        self.NUMBER_Z_SLICES_TO_TRIM=NUMBER_Z_SLICES_TO_TRIM
        self.CLUSTER_RADIUS=CLUSTER_RADIUS
        self.list_thresholds_spot_detection=list_thresholds_spot_detection
        self.list_average_spots_per_cell=list_average_spots_per_cell
        self.list_number_detected_cells=list_number_detected_cells
        self.list_is_image_sharp=list_is_image_sharp
        self.list_metric_sharpeness_images=list_metric_sharpeness_images
        self.remove_out_of_focus_images=remove_out_of_focus_images
        self.sharpness_threshold=sharpness_threshold
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
            list_processing_image=[]
            list_image_id =[]
            with open(filename, 'w') as fd:
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nAUTHOR INFORMATION  ')
                fd.write('\n    Author: ' + getpass.getuser())
                fd.write('\n    Created: ' + datetime.datetime.today().strftime('%d %b %Y'))
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
                    if not(self.threshold_for_spot_detection in (None, [None]) ):
                        fd.write('\n        threshold_spot_detection: ' + str(self.threshold_for_spot_detection[k]) )
                    else:
                        fd.write('\n        threshold_spot_detection: ' + 'automatic value using BIG-FISH' )
                fd.write('\n    minimum_spots_cluster: ' + str(self.minimum_spots_cluster) )
                fd.write('\n    remove_z_slices_borders: ' + str(self.remove_z_slices_borders) )
                fd.write('\n    number of z-slices trimmed at each border: ' + str(self.NUMBER_Z_SLICES_TO_TRIM) )
                fd.write('\n    cluster radius: ' + str(self.CLUSTER_RADIUS) )
                fd.write('\n    remove_out_of_focus_images: ' + str(self.remove_out_of_focus_images))
                if self.remove_out_of_focus_images == True:
                    fd.write('\n    sharpness_threshold: ' + str(self.sharpness_threshold))
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nFILES AND DIRECTORIES USED ')
                fd.write('\n    Directory path: ' + str(self.data_dir) )
                fd.write('\n    Folder name: ' + str(self.data_dir.name)  )
                # for loop for all the images.
                fd.write('\n    Images in the directory :'  )
                # size of longest name string
                file_name_len =0
                max_file_name_len =0
                for _, img_name in enumerate (self.list_files_names):
                    if len(img_name) > file_name_len:
                        max_file_name_len = len(img_name)
                    else:
                        max_file_name_len =0
                
                str_label_img = '| Image Name'
                size_str_label_img = len(str_label_img)
                space_for_image_name = np.min((size_str_label_img, (size_str_label_img-max_file_name_len)))+1
                fd.write('\n        '+ str_label_img+' '* space_for_image_name + '      '+ '| Sharpness metric' + '      ' +'| Image Id'  )
                counter=0
                for indx, img_name in enumerate (self.list_files_names):
                    file_name_len = len(img_name)
                    difference_name_len = max_file_name_len-file_name_len
                    if (self.list_segmentation_successful[indx]== True) and (self.list_is_image_sharp[indx]== True):
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4))+ ' '*8+ str(self.list_metric_sharpeness_images[indx]) +  '        ' + str(self.list_counter_image_id[counter]) )
                        list_image_id.append(self.list_counter_image_id[counter])
                        counter+=1
                        list_processing_image.append('successful')
                    elif self.list_is_image_sharp[indx]== False:
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4)) + ' '*8 + str(self.list_metric_sharpeness_images[indx])+ '      - error out of focus.')
                        list_processing_image.append('error out of focus')
                        list_image_id.append(-1)
                    else:
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4))+ ' '*8 + str(self.list_metric_sharpeness_images[indx])+ '      - error segmentation.')
                        list_processing_image.append('error segmentation')
                        list_image_id.append(-1)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nSUMMARY RESULTS')
                # iterate for all processed images and printing the obtained threshold intensity value.
                for k in range(len(self.channels_with_FISH)):
                    fd.write('\n    For Channel ' + str(self.channels_with_FISH[k]) )
                    fd.write('\n             Image Id    |    threshold    |    number cells    |  mean spots per cell |' )
                    for i,image_id in enumerate(self.list_counter_image_id) :
                        image_id_str = str(image_id)
                        len_id = len(image_id_str)
                        threshold_str = str(int(self.list_thresholds_spot_detection[i][k]))
                        len_ts= len(threshold_str)
                        number_cells_str = str(int(self.list_number_detected_cells[i]))
                        len_nc = len(number_cells_str)
                        average_spots_per_cells_str = str(int(self.list_average_spots_per_cell[i][k]))
                        #len_sp = len(average_spots_per_cells_str)
                        fd.write('\n                ' +'    '+image_id_str + ' '* np.max((1,(13-len_id))) +
                                                '    '+threshold_str +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+number_cells_str + ' '* np.max((1,(17-len_nc))) +
                                                '    '+average_spots_per_cells_str )
                    
                    total_average_number_cells = str(int(np.mean(self.list_number_detected_cells)))
                    total_detected_cells = str(int(np.sum(self.list_number_detected_cells)))
                    #total_average = ' '#str(int(np.mean(self.list_average_spots_per_cell[:][k])))
                    
                    fd.write('\n              ' +'Average:' + ' '* np.max((1,(13-len_id))) +
                                                '    '+' '*len_ts +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+ total_average_number_cells + ' '* np.max((1,(17-len_nc))) )
                    
                    fd.write('\n              ' +'Total  :' + ' '* np.max((1,(13-len_id))) +
                                                '    '+' '*len_ts +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+ total_detected_cells + ' '* np.max((1,(17-len_nc)))  )
                    
                        #self.list_average_spots_per_cell, self.list_number_detected_cells
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nREPRODUCIBILITY ')
                fd.write('\n    Platform: \n')
                fd.write('        Python: ' + str(platform.python_version()) )
                fd.write('\n    Dependencies: ')
                # iterating for all modules
                for module_name in installed_modules:
                    if any(module_name[0:4] in s for s in important_modules):
                        fd.write('\n        '+ module_name)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
            return list_processing_image,list_image_id
        create_data_file(self.filename)
        list_processing_image, list_image_id = write_data_in_file(self.filename)
        
        data = {'Image_id': list_image_id, 'Image_name': self.list_files_names, 'Processing': list_processing_image}
        df = pd.DataFrame(data)
        df.to_csv(self.filename_csv)
        
        return None

