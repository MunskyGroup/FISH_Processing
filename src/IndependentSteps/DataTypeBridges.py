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

import h5py
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import IndependentStepClass, DataContainer
from src.Parameters import Parameters
from src.Util import Utilities, NASConnection
from src.GeneralOutput import OutputClass


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

# Loading in data from multiple locations
class New_Parameters(OutputClass):
    def append(self, new_params):
        Parameters.update_parameters(new_params)


class DataTypeBridge(IndependentStepClass):
    def main(self, initial_data_location, connection_config_location, 
            load_in_mask, nucChannel, cytoChannel, 
            independent_params, local_dataset_location: list[str] = None,
            **kwargs):
        # save to a single location
        database_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        database_loc = os.path.join(database_loc, 'dataBases')
        # if data is local
        if local_dataset_location is not None:
            if isinstance(local_dataset_location, str):
                local_dataset_location = [local_dataset_location]
            
            # if data is h5 format already
            if local_dataset_location[0].endswith('h5'):
                h5_names = [os.path.basename(f) for f in local_dataset_location]
                folders = [os.path.dirname(f) for f in local_dataset_location]

            # if data is hiding in a folder
            elif os.path.isdir(local_dataset_location[0]):
                h5_names = []
                for ds in local_dataset_location:
                    if os.path.isdir(ds):
                        h5_name = os.path.basename(ds) + '.h5'
                        h5_names.append(h5_name)
                        self.convert_folder_to_H5(ds, h5_name, nucChannel, cytoChannel)
                folders = local_dataset_location
                
        # if data is on originates from the nas
        else:
            if type(initial_data_location) == str:
                initial_data_location = [initial_data_location]
            names = [os.path.basename(location) for location in initial_data_location]
            folders = [os.path.join(database_loc, n) for n in names]
            h5_names = [n + '.h5' for n in names]
            for i, location in enumerate(initial_data_location):
                folder = folders[i]
                h5_name = h5_names[i]
                self.download_folder_from_NAS(location, folder, connection_config_location)
                self.convert_folder_to_H5(folder, h5_name, nucChannel, cytoChannel)

        # Load in H5 and build independent params
        self.load_in_dataset(folders, h5_names, load_in_mask, independent_params, initial_data_location)

    def download_folder_from_NAS(self, remote_folder_path, local_folder_path, connection_config_location):
        # Downloads a folder from the NAS, confirms that the it has not already been downloaded
        if not os.path.exists(local_folder_path):
            nas = NASConnection(pathlib.Path(connection_config_location))
            os.makedirs(local_folder_path, exist_ok=True)
            nas.copy_folder(remote_folder_path=pathlib.Path(remote_folder_path), 
                        local_folder_path=local_folder_path)

    @abstractmethod
    def convert_folder_to_H5(self, folder, h5_name, nucChannel, cytoChannel):
        # For any standardized data type this will convert it to a h5 file and save that file in the folder
        # that the data originally came from
        ...

    def load_in_dataset(self, locations, H5_names, load_in_mask, independent_params, NAS_locations) -> DataContainer:
        # given an h5 file this will load in the dat in a uniform manner

        H5_locations = [os.path.join(location, H5_name) for location, H5_name in zip(locations, H5_names)]
        position_indexs = []

        h5_files = [h5py.File(H5_location, 'r') for H5_location in H5_locations]
        # TODO: check if another file is already opening the h5 file


        images = [da.from_array(h5['raw_images']) for h5 in h5_files]
        position_indexs = [img.shape[0] for img in images]
        images = da.concatenate(images, axis=0)
        images = images.rechunk((1, 1, -1, -1, -1, -1))
        if load_in_mask:
            masks = [da.from_array(h5['masks']) for h5 in h5_files]
            masks = da.concatenate(masks, axis=0)
            masks = masks.rechunk((1, 1, -1, -1, -1, -1))
        else:
            masks = da.zeros([images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4], images.shape[5]]) # TODO This may give problems in the future but will live with it

        num_chuncks = images.shape[0] * images.shape[1]

        position_indexs = np.cumsum(position_indexs)

        # we have 3 inputs for independent params list[dict], dict, dict w/ proper keys
        if isinstance(independent_params, dict) and set(independent_params.keys()) == set(np.arange(position_indexs[-1]).tolist()):
            # handles dict w/ proper keys
            ip = independent_params
        else:
            if not isinstance(independent_params, list):
                # handles dict w/o proper keys
                independent_params = [independent_params]

            # converts all to proper keys
            ip = {}
            for i, p in enumerate(position_indexs):
                if independent_params is not None and len(independent_params) > 1:
                    independent_params[i]['NAS_location'] = os.path.join(NAS_locations[i], H5_names[i])
                    if i == 0:
                        temp = {p_idx: independent_params[i] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[i] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                elif independent_params is not None and len(independent_params) == 1:
                    independent_params[0]['NAS_location'] = os.path.join(NAS_locations[i], H5_names[i])
                    if i == 0:
                        temp = {p_idx: independent_params[0] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[0] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                else:
                    print('Something is broken')


        DataContainer(local_dataset_location = H5_locations,
                             h5_file = h5_files,
                            total_num_chunks = num_chuncks,
                            images = images,
                            masks = masks)
        
        New_Parameters({'independent_params': ip, 'position_indexs': position_indexs})

    def delete_folder(self, folder):
        shutil.rmtree(folder)


#%% Data Bridges
class NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def convert_folder_to_H5(self, folder, H5_name, nucChannel, cytoChannel):
        pass


class Pycromanager2NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def convert_folder_to_H5(self, folder, H5_name, nucChannel, cytoChannel):
        # check if h5 file already exists
        if os.path.exists(os.path.join(folder, H5_name)):
            return 'already exists'
        
        ds = Dataset(folder)
        
        imgs = ds.as_array('position', 'time', 'channel', 'z', 'y', 'x')

        da.to_hdf5(os.path.join(folder, H5_name), '/raw_images', imgs)


class FFF2NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def convert_folder_to_H5(self, folder, H5_name, nucChannel, cytoChannel): 
        # check if h5 file already exists
        if os.path.exists(os.path.join(folder, H5_name)):
            return 'already exists'
        
        files = os.listdir(folder)
        tifs = [f for f in files if f.endswith('.tif')]
        logs = [f for f in files if f.endswith('.log')]
        mask_dirs = [f for f in files if f.startswith('masks')]

        already_made_masks = False

        if len(mask_dirs) > 0:
            zipped_mask_dir = [f for f in mask_dirs if f.endswith('.zip')]

            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            if len(zipped_mask_dir) == 1 and len(mask_tifs) == 0:
                shutil.unpack_archive(os.path.join(folder, zipped_mask_dir[0]), folder)
                already_made_masks = True
            
            mask_dirs = [f for f in files if f.startswith('masks')]
            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            mask_cells = [f for f in mask_tifs if 's_cyto_R' in f]
            mask_nuclei = [f for f in mask_tifs if 'masks_nuclei_R' in f]
            already_made_masks = True
    
        # create list of images
        list_images_names = [f for f in tifs if not f.startswith('masks')]
        list_channels = np.sort(list(set([f.split('_')[-1].split('.')[0] for f in list_images_names])))
        list_roi = np.sort(list(set([f.split('_')[0] for f in list_images_names])))
        timepoints = np.sort(list(set([f.split('_')[3] for f in list_images_names])))

        number_of_timepoints = len(set(timepoints))
        number_color_channels = len(set(list_channels))
        number_of_fov = len(set(list_roi))

        # os.makedirs(local_folder, exist_ok=True)
        imgs = None
        masks = None
        count = 0
        img_metadata = {}
        for t in range(number_of_timepoints):
            tp = timepoints[t]
            for r in range(number_of_fov):
                fov = list_roi[r]

                for c in range(number_color_channels):
                    channel = list_channels[c]
                    search_params = [fov, channel, tp]
                    img_name = [f for f in list_images_names if all(v in f for v in search_params)][0]
                    img = tifffile.imread(os.path.join(folder, img_name))
                    img = da.from_array(img)
                    # make all the image data floats
                    img = img.astype(np.float32)

                    search_params = [fov]
                    log_name = [f for f in logs if all(v in f for v in search_params)][0]
                    with open(os.path.join(folder, log_name), 'r') as f:
                        log = f.readlines()

                    if fov not in img_metadata:
                        img_metadata[fov] = {}
                    img_metadata[fov][tp] = log

                    if imgs is None:
                        imgs = da.zeros((number_of_fov, number_of_timepoints, number_color_channels, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

                    if masks is None:
                        masks = da.zeros((number_of_fov, 1, number_color_channels, 1, img.shape[1], img.shape[2]), dtype=np.float32)

                    imgs[r, t, c, :, :, :] = img

                    if already_made_masks:
                        search_params = [fov, tp]
                        cell_mask_name = [f for f in mask_cells if all(v in f for v in search_params)][0] if len(mask_cells) > 0 else None
                        nuc_mask_name = [f for f in mask_nuclei if all(v in f for v in search_params)][0] if len(mask_nuclei) > 0 else None
                        if cell_mask_name is not None and cytoChannel is not None:
                            # print('cell ', cell_mask_name)
                            mask = tifffile.imread(os.path.join(folder, cell_mask_name))
                            masks[r, 0, cytoChannel,0, :, :] = da.from_array(mask)
                        if nuc_mask_name is not None and nucChannel is not None:
                            # print('nuc ', nuc_mask_name)
                            mask = tifffile.imread(os.path.join(folder, nuc_mask_name))
                            masks[r, 0, nucChannel, 0, :, :] = da.from_array(mask)
                    count += 1

        imgs = imgs.rechunk((1, 1, -1, -1, -1, -1))
        masks = masks.rechunk((1, 1, -1, -1, -1, -1))
        
        da.to_hdf5(os.path.join(folder, H5_name), {'/raw_images': imgs, '/masks': masks}, compression='gzip')

        metadata_str = json.dumps(img_metadata)
        with h5py.File(os.path.join(folder, H5_name), 'a') as h5f:
            if '/metadata' in h5f:
                del h5f['/metadata']
            h5f.create_dataset('/metadata', data=metadata_str)

        del imgs
        del masks
        del metadata_str
                

        # save the data to a NDTIFF Dataset





#%% Auxilary Functions
class Avg_Parameters(IndependentStepClass):
    def main(self, dataset_to_avg: list[str], previous_analysis_name: str, params_to_avg: list[str], local_dataset_location:str, h5_file,
              **kwargs):

        # find the local location of the dataset
        ds_names = [os.path.basename(dataset) for dataset in dataset_to_avg]

        # find h5 files in the directory
        h5_files = [f for f in os.listdir(ds_names[0]) if f.endswith('.h5')]

        # Load in each h5 file and find the group with previous analysis name
        # then in that group find the parameters to average
        # then average those parameters
        values_to_average = np.empty((len(params_to_avg), len(dataset_to_avg)))
        for h, h5 in enumerate(h5_files):
            path = os.path.join(h5, f'{h5}.h5')
            if h5 in local_dataset_location:
                if previous_analysis_name in h5_file.keys():
                        for p, param in enumerate(params_to_avg):
                            if param in h5_file[previous_analysis_name].keys():
                                values_to_average[p, h] = h5_file[previous_analysis_name][param]
            else:
                with h5py.File(h5, 'r') as f:
                    if previous_analysis_name in f.keys():
                        for p, param in enumerate(params_to_avg):
                            if param in f[previous_analysis_name].keys():
                                values_to_average[p, h] = f[previous_analysis_name][param]
        
        # average the values
        avg_values = np.mean(values_to_average, axis=1)

        # make this to a dictionary
        avg_params = dict(zip(params_to_avg, avg_values))

        return New_Parameters(avg_params)
