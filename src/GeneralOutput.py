from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
import h5py
import tables
import pandas as pd

# Many of the output classes will be the same, so we can create a base class and then inherit from it
# however, they will have differences on if they modify a PipelineDataClass or if they are the final output

# The Step Classes will be similar however they will have differences on the inputs they act on 
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
    def save_all_outputs(cls, location, h5_file: str, group_name: str):
        # get all the instances of the class
        instances = cls.get_all_instances()
        # save them to the h5 file
        for i, instance in enumerate(instances):
            instance.save(location, h5_file, group_name)

    @abstractmethod
    def append(self, *args, **kwargs):
        pass

    def save(self, location, h5_file: str, group_name: str):
        # get all the attributes of the class
        attributes = vars(self)

        def handle_df(df):
            for col in df.columns:
                if df[col].dtype == 'O':  # Object type
                    if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
                        df[col] = df[col].astype(str)  # Convert to string
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, if possible
            return df

        h5_file.close()

        # save them to the h5 file 

        h5_file = h5py.File(location, 'a')
        
        # check if the group exists
        if group_name in h5_file:
            group = h5_file[group_name]
        else:
            group = h5_file.create_group(group_name)
            
        for key in attributes:
            if key != '_initialized':
                data = attributes[key]
                if data is not None:
                    if type(data) == pd.DataFrame:
                        data = handle_df(data)

                
                    # if dataset is already made, delete it
                    if key in group:
                        del group[key]

                    group.create_dataset(key, data=data)

        h5_file.close()


        




