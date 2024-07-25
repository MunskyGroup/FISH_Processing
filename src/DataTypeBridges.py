import pathlib
import pycromanager as pycro
import shutil
import tifffile
import numpy as np

from src.Util.Utilities import Utilities
# from .Util import Utilities
from . import Experiment, PipelineSettings, ScopeClass



class Pycromanager2NativeDataType():
    def __init__(self, experiment:Experiment, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, connection_config_location):
        self.experiment = experiment
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        
        self.local_data_dir, self.masks_dir, self.list_files_names, self.list_images_all_fov, self.list_images, \
        self.number_of_fov, self.number_color_channels, self.number_z_slices, self.number_of_timepoints, self.list_tps, self.list_nZ, self.number_of_imgs = self.convert_to_standard_format(data_folder_path=pathlib.Path(experiment.initial_data_location), 
                                                                                                                                                            path_to_config_file=connection_config_location, 
                                                                                                                                                            download_data_from_NAS = pipelineSettings.local_or_NAS,
                                                                                                                                                            use_metadata=1,
                                                                                                                                                            is_format_FOV_Z_Y_X_C=1)
                                                        

    def convert_to_standard_format(self, data_folder_path, path_to_config_file, download_data_from_NAS, use_metadata, is_format_FOV_Z_Y_X_C):
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
                # print(metadata.axes)
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
            counter = 0
            list_images_standard_format = []
            list_files_names = []
            list_tps = []
            list_zs = []
            number_of_imgs = number_of_fov*number_of_tp
            number_of_files = len(list_files_names_all_fov)
            number_of_X = None
            number_of_Y = None
            for i in range(number_of_files):
                for tp in range(number_of_tp):
                    for fov in range(number_of_fov):
                        if not (number_of_X is None):
                            temp_image = np.zeros((number_z_slices, number_of_Y, number_of_X, number_color_channels))
                        for z in range(number_z_slices):
                            for c in range(number_color_channels):
                                if number_of_X is None:
                                    temp_image = metadata.read_image(position=fov, time=tp, z=z, channel=c)
                                    number_of_X = temp_image.shape[1]
                                    number_of_Y = temp_image.shape[0]
                                    temp_image = np.zeros((number_z_slices, number_of_Y, number_of_X, number_color_channels))
                                temp_image[z, :, :, c] = metadata.read_image(position=fov, time=tp, z=z, channel=c)
                                list_tps.append(tp)
                                list_zs.append(z)

                        list_images_standard_format.append(temp_image)
                        list_files_names.append(  list_files_names_all_fov[i].split(".")[0]+'_tp_'+str(tp)+'_fov_'+str(fov)+'.tif' )
                        tifffile.imsave(str(destination_folder.joinpath(list_files_names[-1])), list_images_standard_format[-1])
                        counter+=1
        masks_dir = None
        return destination_folder,masks_dir, list_files_names, list_images_all_fov, list_images_standard_format, number_of_fov, number_color_channels, number_z_slices, number_of_tp, list_tps, list_zs, number_of_imgs
        




        