import pathlib
import pycromanager as pycro
import shutil
import tifffile
import numpy as np
import os
import sys
import inspect
from datetime import datetime
from abc import ABC, abstractmethod
import dask_image.imread as dask_imread
from dask import array as da
from ndstorage import NDTiffDataset, NDTiffPyramidDataset
from ndtiff import Dataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import IndependentStepClass, DataContainer, Parameters
from src.Util import Utilities, NASConnection


#%% Useful Functions
def get_first_executing_folder():
    # Get the current stack
    stack = inspect.stack()
    
    # The first frame in the stack is the current function,
    # The second frame is the caller. However, we want the first script.
    # To find that, we look for the first frame that isn't from an internal call.
    for frame in stack:
        if frame.filename != __file__:
            # Extract the directory of the first non-internal call
            return os.path.dirname(os.path.abspath(frame.filename))

    return None

#%% Abstract Class
class DataTypeBridge(IndependentStepClass):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def main(self, initial_data_location, connection_config_location, 
             download_data_from_NAS, load_in_mask, index_dict: dict = None, 
             **kwargs):
        pass

    def download_folder_from_NAS(self, remote_folder_path, local_folder_path, connection_config_location, download_data_from_NAS):
        if not os.path.exists(local_folder_path) and download_data_from_NAS:
            nas = NASConnection(pathlib.Path(connection_config_location))
            os.makedirs(local_folder_path, exist_ok=True)   
            nas.copy_folder(remote_folder_path=pathlib.Path(remote_folder_path), 
                        local_folder_path=local_folder_path)

    def convert_folder_to_NDTIFF(self, local_folder):
        pass

    def load_in_dataset(self, local_folder_path, load_in_mask, index_dict) -> DataContainer:
        ds = pycro.Dataset(local_folder_path)
        images = ds.as_array()
        if load_in_mask:
            # find files in folder containing 'masks'
            mask_files = [f for f in os.listdir(local_folder_path) if 'masks' in f]
            if len(mask_files) == 0:
                print('No mask files found in folder')

            if len(mask_files) > 1:
                raise ValueError('Multiple mask files found in folder. Please ensure only one mask file is present')
            
            else:
                masks = dask_imread.imread(os.path.join(local_folder_path, mask_files[0]))



        # get the experiment params
        experiment = None
        for instance in Parameters.get_parameters():
            if instance.__class__.__name__ == 'Experiment':
                experiment = instance
                break
        if experiment is None:
            raise ValueError('Experiment class not found in parameters')
        
        if index_dict is None:
            # find the axes of the dataset
            axes = ds.axes()
            if 'z' in axes:
                # find where the len of that axes is equal to the shape of the image
                z_axis = [i for i, ax in enumerate(axes) if len(ax) == images.shape[index_dict['z']]]
                if len(z_axis) == 0:
                    raise ValueError('No z axis found in the dataset')
                if len(z_axis) > 1:
                    raise ValueError('Cannot destinguish between multiple z axes')
                z_axis = z_axis[0]
            if 'time' in axes:
                time_axis = [i for i, ax in enumerate(axes) if len(ax) == images.shape[index_dict['t']]]
                if len(time_axis) == 0:
                    raise ValueError('No time axis found in the dataset')
                if len(time_axis) > 1:
                    raise ValueError('Cannot destinguish between multiple time axes')
                time_axis = time_axis[0]
            if 'channel' in axes:
                channel_axis = [i for i, ax in enumerate(axes) if len(ax) == images.shape[index_dict['c']]]
                if len(channel_axis) == 0:
                    raise ValueError('No channel axis found in the dataset')
                if len(channel_axis) > 1:
                    raise ValueError('Cannot destinguish between multiple channel axes')
                channel_axis = channel_axis[0]
            if 'position' in axes:
                position_axis = [i for i, ax in enumerate(axes) if len(ax) == images.shape[index_dict['p']]]
                if len(position_axis) == 0:
                    raise ValueError('No position axis found in the dataset')
                if len(position_axis) > 1:
                    raise ValueError('Cannot destinguish between multiple position axes')
                position_axis = position_axis[0]

            index_dict = {'z': z_axis, 't': time_axis, 'c': channel_axis, 'p': position_axis, 'y': -2, 'x': -1}
            
        # this is gonna be the product of the shape of the image at z, c, t, p
        num_z = images.shape[index_dict['z']] if 'z' in index_dict else 1
        num_c = images.shape[index_dict['c']] if 'c' in index_dict else 1
        num_t = images.shape[index_dict['t']] if 't' in index_dict else 1
        num_p = images.shape[index_dict['p']] if 'p' in index_dict else 1
        total_num_chuncks = num_z * num_c * num_t * num_p

        data = DataContainer(local_folder_path, total_num_chuncks, images, ds, masks)
        return data
        
    def delete_folder(self, folder):
        shutil.rmtree(folder)


#%% Data Bridges
class Pycromanager2NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def main(self, initial_data_location, connection_config_location, 
             download_data_from_NAS, load_in_mask, index_dict, **kwargs):
        local_folder_path = 'Analysis_' + os.path.basename(local_folder_path) + '_' + datetime.now().strftime('%Y-%m-%d')
        self.download_folder_from_NAS(initial_data_location, local_folder_path, connection_config_location, download_data_from_NAS)
        self.load_in_dataset(local_folder_path, load_in_mask, index_dict)


class FFF2NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def main(self, initial_data_location, connection_config_location, 
             cytoChannel, nucChannel, download_data_from_NAS, load_in_mask, 
             index_dict: dict = None, **kwargs):
        temp_folder = 'temp_' + os.path.basename(initial_data_location)
        local_folder_path = 'Analysis_' + os.path.basename(initial_data_location) + '_' + datetime.now().strftime('%Y-%m-%d')
        self.download_folder_from_NAS(initial_data_location, temp_folder, connection_config_location, download_data_from_NAS)

        self.convert_folder_to_NDTIFF(temp_folder, local_folder_path, nucChannel, cytoChannel)
        self.delete_folder(temp_folder)

        self.load_in_dataset(local_folder_path, load_in_mask, index_dict)

    def convert_folder_to_NDTIFF(self, temp_folder, local_folder, nucChannel, cytoChannel): 
        files = os.listdir(temp_folder)
        tifs = [f for f in files if f.endswith('.tif')]
        logs = [f for f in files if f.endswith('.log')]
        mask_dirs = [f for f in files if f.startswith('masks')]

        already_made_masks = False

        if len(mask_dirs) > 0:
            unzipped_mask_dir = [f for f in mask_dirs if os.path.isdir(os.path.join(temp_folder, f))]
            zipped_mask_dir = [f for f in mask_dirs if f.endswith('.zip')]

            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            if len(zipped_mask_dir) == 1 and len(mask_tifs) == 0:
                shutil.unpack_archive(os.path.join(temp_folder, zipped_mask_dir[0]), temp_folder)
                already_made_masks = True
            
            mask_dirs = [f for f in files if f.startswith('masks')]
            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            mask_cells = [f for f in mask_tifs if 's_cyto_R' in f]
            mask_nuclei = [f for f in mask_tifs if 'nuclei' in f]
            mask_cyto = [f for f in mask_tifs if 'cyto_no_nuclei' in f]
            already_made_masks = True
    
        # create list of images
        list_images_names = [f for f in tifs if not f.startswith('masks')]
        list_channels = np.sort(list(set([f.split('_')[-1].split('.')[0] for f in list_images_names])))
        list_roi = np.sort(list(set([f.split('_')[0] for f in list_images_names])))
        list_names = [f.split('_')[1] for f in list_images_names]
        z_slices = np.sort(list(set([f.split('_')[2] for f in list_images_names])))
        timepoints = np.sort(list(set([f.split('_')[3] for f in list_images_names])))

        number_of_timepoints = len(set(timepoints))
        number_z_slices = len(set(z_slices))
        number_color_channels = len(set(list_channels))
        number_of_fov = len(set(list_roi))

        number_of_images_to_process = number_of_fov * number_of_timepoints

        os.makedirs(local_folder, exist_ok=True)
        imgs = None
        masks = None
        count = 0
        for t in range(number_of_timepoints):
            tp = timepoints[t]
            for r in range(number_of_fov):
                fov = list_roi[r]

                for c in range(number_color_channels):
                    channel = list_channels[c]
                    search_params = [fov, channel, tp]
                    img_name = [f for f in list_images_names if all(v in f for v in search_params)][0]
                    img = tifffile.imread(os.path.join(temp_folder, img_name))
                    img = da.from_array(img)
                    # make all the image data floats
                    img = img.astype(np.float32)

                    search_params = [fov]
                    log_name = [f for f in logs if all(v in f for v in search_params)][0]
                    with open(os.path.join(temp_folder, log_name), 'r') as f:
                        log = f.readlines()
                    # img_metadata = {'log': log}
                    img_metadata = {'testing': 'bullshit'}

                    if imgs is None:
                        imgs = da.zeros((number_of_fov, number_of_timepoints, number_color_channels, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

                    if masks is None:
                        masks = da.zeros((number_of_fov, number_of_timepoints, number_color_channels, 1, img.shape[1], img.shape[2]), dtype=np.float32)
                        # tp = int(''.join(filter(str.isdigit, tp)))
                        # fov = int(''.join(filter(str.isdigit, fov)))
                        # channel = int(''.join(filter(str.isdigit, channel)))
                        # z = z

                        # img_coords = {'time': int(tp), 'channel': int(channel), 'position': int(fov), 'z': int(z)}
                        

                    imgs[r, t, c, :, :, :] = img

                    search_params = [fov, tp]
                    if already_made_masks:
                        cell_mask_name = [f for f in mask_cells if all(v in f for v in search_params)][0] if len(mask_cells) > 0 else None
                        cyto_mask_name = [f for f in mask_cyto if all(v in f for v in search_params)][0] if len(mask_cyto) > 0 else None
                        nuc_mask_name = [f for f in mask_nuclei if all(v in f for v in search_params)][0] if len(mask_nuclei) > 0 else None
                        if cell_mask_name is not None:
                            masks[r, t, cytoChannel, :, :, :] = da.from_array(tifffile.imread(os.path.join(temp_folder, cell_mask_name)))
                        if nuc_mask_name is not None:
                            masks[r, t, nucChannel, :, :, :] = da.from_array(tifffile.imread(os.path.join(temp_folder, nuc_mask_name)))
                    count += 1

        # # save dask arrays
        # imgs = imgs
        # masks = masks.compute()

        da.to_hdf5(os.path.join(local_folder, 'data.hdf5'), '/images', imgs)
        da.to_hdf5(os.path.join(local_folder, 'data.hdf5'), '/masks', masks)

                

        # save the data to a NDTIFF Dataset



if __name__ == '__main__':
    from src import Experiment, Settings, ScopeClass, DataContainer
    experiment = Experiment(nucChannel=0, cytoChannel=1)
    settings = Settings()
    scope = ScopeClass()
    data = DataContainer()

    experiment.initial_data_location = r'smFISH_images\Eric_smFISH_images\20230511\DUSP1_DexTimeConcSweep_10nM_75min_041223'

    FFF2NativeDataType().run()




  