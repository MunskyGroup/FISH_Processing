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

    def main(self, initial_data_location, connection_config_location, 
             download_data_from_NAS, load_in_mask, nucChannel, cytoChannel, index_dict: dict = None, 
             **kwargs):
        h5_name = os.path.basename(initial_data_location) + '.h5'
        folder = os.path.basename(initial_data_location)
        self.download_folder_from_NAS(initial_data_location, folder, connection_config_location, download_data_from_NAS)
        self.convert_folder_to_H5(folder, h5_name, nucChannel, cytoChannel)
        self.load_in_dataset(folder, h5_name, load_in_mask)

    def download_folder_from_NAS(self, remote_folder_path, local_folder_path, connection_config_location, download_data_from_NAS):
        if not os.path.exists(local_folder_path) and download_data_from_NAS:
            nas = NASConnection(pathlib.Path(connection_config_location))
            os.makedirs(local_folder_path, exist_ok=True)
            nas.copy_folder(remote_folder_path=pathlib.Path(remote_folder_path), 
                        local_folder_path=local_folder_path)

    @abstractmethod
    def convert_folder_to_H5(self, folder, h5_name, nucChannel, cytoChannel):
        ...

    def load_in_dataset(self, location, H5_name, load_in_mask) -> DataContainer:
        H5_location = os.path.join(location, H5_name)
        f = h5py.File(H5_location, 'r')
        images = da.from_array(f['raw_images'])
        images = images.rechunk((1, 1, -1, -1, -1, -1))
        
        masks = None
        if load_in_mask:
            masks = da.from_array(f['masks'])

        num_chuncks = images.shape[0] * images.shape[1]

        data = DataContainer(local_dataset_location = H5_location,
                            total_num_chunks = num_chuncks,
                            images = images,
                            masks = masks)
        return data
        
    def delete_folder(self, folder):
        shutil.rmtree(folder)


#%% Data Bridges
class Pycromanager2NativeDataType(DataTypeBridge):
    def __init__(self):
        super().__init__()

    def convert_folder_to_H5(self, folder, H5_name, nucChannel, cytoChannel):
        ds = Dataset(folder)
        
        imgs = ds.as_array('position', 'time', 'channel', 'z', 'x', 'y')

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
            mask_nuclei = [f for f in mask_tifs if 'nuclei' in f]
            mask_cyto = [f for f in mask_tifs if 'cyto_no_nuclei' in f]
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

                    search_params = [fov, tp]
                    if already_made_masks:
                        cell_mask_name = [f for f in mask_cells if all(v in f for v in search_params)][0] if len(mask_cells) > 0 else None
                        nuc_mask_name = [f for f in mask_nuclei if all(v in f for v in search_params)][0] if len(mask_nuclei) > 0 else None
                        if cell_mask_name is not None:
                            masks[r, 0, cytoChannel, :, :, :] = da.from_array(tifffile.imread(os.path.join(folder, cell_mask_name)))
                        if nuc_mask_name is not None:
                            masks[r, 0, nucChannel, :, :, :] = da.from_array(tifffile.imread(os.path.join(folder, nuc_mask_name)))
                    count += 1

        da.to_hdf5(os.path.join(folder, H5_name), '/raw_images', imgs)
        da.to_hdf5(os.path.join(folder, H5_name), '/masks', masks)

        metadata_str = json.dumps(img_metadata)
        with h5py.File(os.path.join(folder, H5_name), 'a') as h5f:
            h5f.create_dataset(f'/metadata', data=metadata_str)

                

        # save the data to a NDTIFF Dataset



if __name__ == '__main__':
    from src import Experiment, Settings, ScopeClass, DataContainer, Parameters
    import matplotlib.pyplot as plt
    experiment = Experiment(nucChannel=0, cytoChannel=1)
    settings = Settings(load_in_mask=True)
    scope = ScopeClass()
    data = DataContainer()

    experiment.initial_data_location = r'smFISH_images\Eric_smFISH_images\20230511\DUSP1_DexTimeConcSweep_10nM_75min_041223'

    FFF2NativeDataType().run()

    print(data.images.shape)
    print(data.masks.shape)
    print(data.local_dataset_location)

    plt.imshow(data.images[0, 0, 0, 0, :, :])

    print(Parameters.Parameters.get_parameters())



  