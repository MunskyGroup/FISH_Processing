from dataclasses import dataclass, fields, field
from typing import Union
from abc import ABC, abstractmethod
import pathlib
import numpy as np
import os
from pycromanager import Dataset
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
from dask_image import imread
import h5py
from dataclasses import asdict
import gc
from skimage.io import imsave
import tempfile

@dataclass
class Parameters(ABC):
    """
    This class is used to store the parameters of the pipeline.
    Alls children class will be singletons and will be stored in the _instances list.
    """
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in cls._instances:
            if isinstance(instance, cls):
                return instance
        instance = super().__new__(cls)
        cls._instances.append(instance)
        return instance

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances

    @classmethod
    def clear_instances(cls):
        # Class method to clear all instances of the parent class
        # deletes all instances of the class
        for instance in cls._instances:
            del instance
        cls._instances = []

    @classmethod
    def validate(cls):
        # make sure settings, scope, experiemnt, and datacontainer are all initialized
        if len(cls._instances) < 3 or len(cls._instances) > 4:
            raise ValueError(f"Settings, ScopeClass, and Experiment must all be initialized")
        # makes sure ScopeClass is in _instances
        if not any(isinstance(instance, ScopeClass) for instance in cls._instances):
            raise ValueError(f"ScopeClass must be initialized")
        # makes sure Experiment is in _instances
        if not any(isinstance(instance, Experiment) for instance in cls._instances):
            raise ValueError(f"Experiment must be initialized")
        # makes sure DataContainer is in _instances
        if not any(isinstance(instance, DataContainer) for instance in cls._instances):
            raise ValueError(f"DataContainer must be initialized")
        # makes sure Settings is in _instances
        if not any(isinstance(instance, Settings) for instance in cls._instances):
            raise ValueError(f"Settings must be initialized")

        # Class method to validate all instances of the parent class
        for instance in cls._instances:
            instance.validate_parameters()

    @classmethod
    def initialize_parameters_instances(cls):
        ScopeClass()
        Experiment()
        DataContainer()
        Settings()

    @classmethod
    def update_parameters(cls, kwargs: dict):
        # Class method to update all instances of the parent class
        if kwargs is None:
            return None
        used_keys = []
        for instance in cls._instances:
            for key, value in kwargs.items():
                # check if the key exists in the instance
                if hasattr(instance, key):
                    setattr(instance, key, value)
                    print(f'Overwriting {key} in {instance.__class__.__name__}')
                    used_keys.append(key)
        # if there are any kwargs left, add them to the settings class
        # remove the used keys
        for key in used_keys:
            kwargs.pop(key)
        
        if kwargs:
            print(f'Adding leftover kwargs to Settings')
            # find the settings instance
            for instance in cls._instances:
                if instance.__class__.__name__ == 'Settings':
                    for key, value in kwargs.items():
                        setattr(instance, key, value)
                        print(f'Adding {key} to {instance.__class__.__name__}')

    def validate_parameters(self):
        pass

    def todict(self):
        # Convert all attributes of the instance to a dictionary
        # return {field.name: getattr(self, field.name) for field in fields(self)}
        return {**{field.name: getattr(self, field.name) for field in fields(self)}, **vars(self)}
    
    def reset(self):
        for field in fields(self):
            setattr(self, field.name, field.default)

    def __str__(self):
        string = f'{self.__class__.__name__}:\n'
        for key, value in self.todict().items():
            # make one large string of all the parameters
            string += f'{key}: {value} \n'
        return string

    @classmethod
    def get_parameters(self) -> dict:
        # Get all the parameters of all instances of the class
        params = {}
        for instance in Parameters._instances:
            # check for duplicates and raise an error if found
            duplicate_keys = set(params.keys()).intersection(set(instance.todict().keys()))
            if duplicate_keys:
                if duplicate_keys != {'kwargs'}:
                    raise ValueError(f"Duplicate parameter found: {duplicate_keys}")
            params.update(instance.todict())
        return params


@dataclass
class ScopeClass(Parameters):
    """
    Class to store the parameters of the microscope.
    Attributes:
    voxel_size_yx: int
    psf_z: int
    psf_yx: int

    Default values will be for Terminator Scope
    
    """
    voxel_size_yx: int = 130
    spot_z: int = 500
    spot_yx: int = 360
    microscope_saving_format: str = 'pycromanager'

    def __init__(self, **kwargs):
        if kwargs is not None: # TODO this needs to be moved up but it has issues being in parmaeters class
            for key, value in kwargs.items():
                setattr(self, key, value)


@dataclass
class Experiment(Parameters):
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image
    """
    initial_data_location: Union[str, list[str]] = None
    index_dict: dict = None
    nucChannel: int = None
    cytoChannel: int = None
    FISHChannel: Union[list[int], int] = None
    voxel_size_z: int = 300  # This is voxel
    independent_params: Union[dict, list[dict]] = None
    kwargs: dict = None
    timestep_s: float = None

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def validate_parameters(self):
        print(DataContainer().local_dataset_location)
        if self.initial_data_location is None and DataContainer().local_dataset_location is None:
            raise ValueError("initial_data_location or local_dataset_location must be set")
        
        if self.nucChannel is None:
            print("nucChannel not set")

        if self.cytoChannel is None:
            print("cytoChannel not set")

        if self.FISHChannel is None:
            print("FISHChannel not set")

        if type(self.FISHChannel) is int:
            self.FISHChannel = [self.FISHChannel]

        # make sure each independent parameter has the same keys
        if self.independent_params is not None:
            if type(self.independent_params) is list:
                for i, params in enumerate(self.independent_params):
                    if i == 0:
                        keys = set(params.keys())
                    else:
                        if keys != set(params.keys()):
                            raise ValueError(f"Independent parameters must have the same keys")
            else:
                self.independent_params = [self.independent_params]
        else:
            self.independent_params = [{}]


@dataclass
class DataContainer(Parameters):
    local_dataset_location: Union[list[pathlib.Path], pathlib.Path] = None
    h5_file: h5py.File = None
    total_num_chunks: int = None
    images: da = None
    masks: da = None
    temp_h5: h5py.File = None

    def __init__(self, **kwargs):
        if not hasattr(self, 'temp_name'):
            self.temp_name = f'temp_{np.random.randint(0, 100000)}'

        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __post_init__(self):
        self.save_temp()
        self.load_temp()

    def __setattr__(self, name, value):
        if name == 'total_num_chunks':
            for instance in Parameters._instances:
                if isinstance(instance, Settings):
                    l = [instance.num_chunks_to_run, value]
                    instance.num_chunks_to_run = min(i for i in l if i is not None)

        if name == 'images':
            self._images_modified = True

        if name == 'masks':
            self._masks_modified = True
        super().__setattr__(name, value)


    def save_temp(self):
        if not hasattr(self, 'temp_masks'):
            self.temp_masks = tempfile.TemporaryDirectory()

        if not hasattr(self, 'temp_images'):
            self.temp_images = tempfile.TemporaryDirectory()

        if not hasattr(self, '_images_modified'):
            self._images_modified = False

        if not hasattr(self, '_masks_modified'):
            self._masks_modified = False

        self.images = self.images.compute()
        self.masks = self.masks.compute()
        self.temp_masks.cleanup()
        self.temp_images.cleanup()

        if self.images is not None and self._images_modified:
            self.images = da.rechunk(da.from_array(self.images), (1, 1, -1, -1,-1,-1))
            da.to_npy_stack(self.temp_images.name, self.images) 
            self._images_modified = False

        if self.masks is not None and self._masks_modified:
            self.masks = da.rechunk(da.from_array(self.masks), (1, 1, -1, -1,-1,-1))
            da.to_npy_stack(self.temp_masks.name, self.masks) 
            self._masks_modified = False

        del self.images
        del self.masks
        gc.collect()
        
        self.images = da.from_npy_stack(self.temp_images.name) 
        self.masks = da.from_npy_stack(self.temp_masks.name)


    def load_temp(self):
        # if hasattr(self, 'temp_name'):
        #     if os.path.exists(f'{self.temp_name}_images'):
        #         self.images = da.from_npy_stack(f'{self.temp_name}_images') 
        #         self._images_modified = False
        #     if os.path.exists(f'{self.temp_name}_masks'):
        #         da.from_npy_stack(f'{self.temp_name}_masks')
        #         self._masks_modified = False
        pass

    def delete_temp(self):
        if hasattr(self, 'temp_masks'):
            self.temp_masks.cleanup()
            del self.temp_masks

        if hasattr(self, 'temp_images'):
            self.temp_images.cleanup()
            del self.temp_images

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@dataclass
class Settings(Parameters):
    name: str = None
    return_data_to_NAS: bool = True
    NUMBER_OF_CORES: int = 4
    save_files: bool = True
    num_chunks_to_run: int = 100_000
    download_data_from_NAS: bool = True  # 0 for local, 1 for NAS
    connection_config_location: str = str(os.path.join(repo_path, 'config_nas.yml')) #r"C:\Users\Jack\Desktop\config_nas.yml" # r"/home/formanj/FISH_Processing_JF/FISH_Processing/config.yml"
    share_name: str = 'share'
    display_plots: bool = True
    load_in_mask: bool = False

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        if self.connection_config_location is None:
            self.connection_config_location = str(os.path.join(repo_path, 'config_nas.yml'))

    def validate_parameters(self):
        if self.name is None:
            raise ValueError("Name must be set")


# class GeneratedOutputs(Parameters): # TODO: clear this, I dont think I use it anymore
#     _instance = None

#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             for key, value in kwargs.items():
#                 setattr(self, key, value)

#     def clear_outputs(self):
#         GeneratedOutputs._instances = []

#     def get_outputs(self):
#         from . import OutputClass
#         self.outputs = OutputClass._instances
#         return self.outputs
    
#     def todict(self):
        # for key in self.__dict__.keys():
        #     try:
        #         step_dict = getattr(self.data, key).__dict__
        #         kwargs_data = {**kwargs_data, **step_dict}
        #         kwargs_data.pop(key)
        #     except AttributeError:
        #         pass
        
        # return kwargs_data


#%% Required Params
class required_params(ABC):
    @abstractmethod
    def validate_parameter(self):
        ...


# class Index_Dict(dict, required_params): # TODO Deal with this bs
#     required_keys = {'x', 'y'}
#     optional_keys = {'p', 'z', 'c'}

#     def __init__(self, *args, **kwargs):
#         # check if it none
#         if self is not None:
#             super().__init__(*args, **kwargs)
#             self.validate_parameter()

#     def validate_parameter(self):
#         if not self.required_keys.issubset(self.keys()):
#             raise ValueError(f"Missing required keys: {self.required_keys - set(self.keys())}")
#         if not self.keys().issubset(self.required_keys.union(self.optional_keys)):
#             raise ValueError(f"Invalid keys: {set(self.keys()) - self.required_keys.union(self.optional_keys)}")
        
#     # when the class is called, it will return the dict
#     def __call__(self):
#         return self
    
#     # when the class is updated from None
#     def __setitem__(self, key, value):
#         super().__setitem__(key, value)
#         self.validate_parameter()
