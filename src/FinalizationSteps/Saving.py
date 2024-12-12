import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import dask.array as da
from datetime import datetime
from abc import abstractmethod
import gc

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters
from src.GeneralOutput import OutputClass
from src.Util.Plots import Plots
from src.Util.Metadata import Metadata
from src.Util.ReportPDF import ReportPDF
from src.Util.Utilities import Utilities
from src.Util.NASConnection import NASConnection

def close_h5_files():
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            # check if the file is open
            try:
                if 'temp' not in obj.filename:
                    print(f"Closing {obj.filename}")
                    obj.close()
            except:
                pass

class Saving(FinalizingStepClass):
    @abstractmethod
    def main(self, **kwargs):
        pass

def handle_df(df):
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type
            if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
                df[col] = df[col].astype(str)  # Convert to string
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, if possible
    return df

def handle_dict(d):
    newd = {}
    for key in d.keys():
        if isinstance(d[key], str):
            newd[key] = d[key]
        elif isinstance(d[key], int):
            newd[key] = d[key]
        elif isinstance(d[key], float):
            newd[key] = d[key]
        elif isinstance(d[key], bool):
            newd[key] = d[key]
        elif isinstance(d[key], list):
            newd[key] = d[key]
        else:
            print(f'IDK what to do with {key}: {type(d[key])}')
    return newd



class Save_Outputs(Saving):
    def main(self, **kwargs):
        params = Parameters.get_parameters()

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        independent_params = params['independent_params']
        position_indexs = params['position_indexs']

        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        OutputClass.save_all_outputs(local_dataset_location, h5_file, f'Analysis_{Analysis_name}_{date}', position_indexs, independent_params)


class Save_Parameters(Saving):
    def main(self, **kwargs):
        def recursively_save_dict_contents_to_group(h5file, path, dic):
            for key, item in dic.items():
                if isinstance(item, dict):
                    recursively_save_dict_contents_to_group(h5file, f"{path}/{key}", item)
                else:
                    h5file[f"{path}/{key}"] = item

        params = Parameters.get_parameters()
        params_to_ignore = ['h5_file', 'local_dataset_location', 'images', 'masks']

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']

        close_h5_files()

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        for i, locaction in enumerate(local_dataset_location):
            # save the parameters to the h5 file
            h5_file = h5py.File(locaction, 'r+')
            group_name = f'Analysis_{Analysis_name}_{date}'
            if group_name in h5_file:
                group = h5_file[group_name]
            else:
                group = h5_file.create_group(group_name)

            # save the parameters to the h5 file
            # remove params_to_ignore
            for key in params_to_ignore:
                if key in params:
                    del params[key]
            
            params = handle_dict(params)

            # if dataset is already made, delete it
            if 'parameters' in group:
                del group['parameters']

            recursively_save_dict_contents_to_group(group, 'parameters', params)

            # save the nuc channel and cyto channel and fish channel at the top level
            if 'nucChannel' in params:
                group.attrs['nucChannel'] = params['nucChannel']
            if 'cytoChannel' in params:
                group.attrs['cytoChannel'] = params['cytoChannel']
            if 'FISHChannel' in params:
                group.attrs['FISHChannel'] = params['FISHChannel']
        
            h5_file.close()


class Save_Images(Saving):
    def main(self, **kwargs):
        params = Parameters.get_parameters()

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        images = params['images']
        position_indexs = params['position_indexs']

        close_h5_files()

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        # save the images to the h5 file
        for i, locaction in enumerate(local_dataset_location):
            if h5_file[i]:
                h5_file[i].close()
            h5 = h5py.File(locaction, 'r+')
            group_name = f'Analysis_{Analysis_name}_{date}'
            if group_name in h5:
                group = h5[group_name]
            else:
                group = h5.create_group(group_name)

            # if dataset is already made, delete it
            if 'images' in group:
                del group['images']

            # save the images to the h5 file
            group.create_dataset('images', data=images[position_indexs[i-1] if i > 0 else 0:position_indexs[i]])

            h5.close()


class Save_Masks(Saving):
    def main(self, **kwargs):
        params = Parameters.get_parameters()

        local_dataset_location = params['local_dataset_location']
        masks = params['masks']
        h5_file = params['h5_file']
        position_indexs = params['position_indexs']

        computed_masks = masks.compute()

        close_h5_files()

        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        if masks is not None:
            for i, locaction in enumerate(local_dataset_location):
                with h5py.File(locaction, 'r+') as h5:
                    # h5 = h5py.File(locaction, 'a')
                    
                    # check if the dataset is already made
                    if '/masks' in h5:
                        del h5['/masks']

                    chunk_size = (1,) + computed_masks.shape[1:]  # Define chunk size
                    h5.create_dataset('/masks', data=computed_masks[position_indexs[i-1] if i > 0 else 0:position_indexs[i]], chunks=chunk_size, compression="gzip")
                    







