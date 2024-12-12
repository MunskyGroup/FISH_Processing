from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
import h5py
import tables
import pandas as pd
import gc

# Many of the output classes will be the same, so we can create a base class and then inherit from it
# however, they will have differences on if they modify a PipelineDataClass or if they are the final output

# The Step Classes will be similar however they will have differences on the inputs they act on 

def close_h5_files():
    for obj in gc.get_objects():

            # check if the file is open
            try:
                if isinstance(obj, h5py.File):
                    if 'temp' not in obj.filename:
                        print(f"Closing {obj.filename}")
                        obj.close()
            except:
                pass

class OutputClass(ABC):
    """ This class will be used to generate singletons for the output classes. """
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in OutputClass._instances:
            if isinstance(instance, cls):
                instance.append(*args, **kwargs)
                return instance
        instance = super().__new__(cls)
        OutputClass._instances.append(instance)
        return instance
    
    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.append(*args, **kwargs)

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances
    
    @classmethod
    def clear_instances(cls):
        # del all instances out of memory
        for instance in cls._instances:
            del instance
        cls._instances = []

    @classmethod
    def save_all_outputs(cls, location, h5_file: str, group_name: str, position_indexs: list[int], independent_params: Dict[str, Any]):
        # get all the instances of the class
        instances = cls.get_all_instances()
        # save them to the h5 file
        for i, instance in enumerate(instances):
            instance.save(location, h5_file, group_name, position_indexs, independent_params)

    @abstractmethod
    def append(self, *args, **kwargs):
        pass

    def save(self, locations, h5_file: str, group_name: str, position_indexs: list[int], independent_params: Dict[str, Any] ):
        # get all the attributes of the class
        attributes = vars(self)

        def handle_df(df):
            for col in df.columns:
                if df[col].dtype == 'O':  # Object type
                    if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
                        df[col] = df[col].astype(str)  # Convert to string
                    else:
                        df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to numeric, if possible
            
            # add the independent params to the dataframe
            if 'fov' in df.columns:
                if independent_params is not None:
                    for name in independent_params[0].keys():
                        if name in df.columns:
                            pass
                        else:
                            df[name] =[independent_params[fov][name] for fov in df['fov'].astype(int).tolist()]
                df = df.sort_values(by='fov')
                df['fov'] = pd.Categorical(df['fov']).codes
            return df
        
        def split_df(df, lower, upper):
            if 'fov' not in df.columns:
                return df
            else:
                positions = df['fov'].unique()
                positions = positions[positions >= lower]
                positions = positions[positions <= upper]

                df = df[df['fov'].isin(positions)]
            return df
        
        close_h5_files()
        
        for i, location in enumerate(locations):
            for key in attributes:
                if key not in ['_initialized', 'independent_params']:
                    data = attributes[key]
                    if data is not None:

                        if isinstance(data, pd.DataFrame):
                            data = handle_df(data)
                            data = split_df(data, position_indexs[i-1] if i > 0 else 0, position_indexs[i])
                            data.to_hdf(location, f'{group_name}/{key}', mode='a', format='table', data_columns=True)

                        else:
                            h5_file = h5py.File(location, 'a')

                            if group_name in h5_file:
                                group = h5_file[group_name]
                            else:
                                group = h5_file.create_group(group_name)

                            if key in group:
                                del group[key]
                            
                            group.create_dataset(key, data=data)

                            h5_file.close()


        




