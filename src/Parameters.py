from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
import pathlib
import numpy as np
import os

@dataclass
class Parameters(ABC):
    """
    This class is used to store the parameters of the pipeline.
    """
    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def pipeline_init(self):
        pass

    @abstractmethod
    def pipeline_init(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def validate_parameters(self):
        pass

    def get_parameters(self):
        return self.__dict__

@dataclass
class ScopeClass:
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
    kwargs: dict = None

    def __init__(self, **kwargs):
        # Loop over all fields defined in the dataclass
        for f in fields(self):
            # Set the attribute with the value from kwargs, or the default if not provided
            setattr(self, f.name, kwargs.get(f.name, f.default))

    def __post_init__(self):
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__


@dataclass
class Experiment(Parameters):
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image


    """
    initial_data_location: str = field(default=None, repr=False)
    number_of_images_to_process: int = None  # This will be all images to process and will be the product of the number of tp and number of FOVs
    number_of_channels: int = None
    number_of_timepoints: int = None
    number_of_Z: int = None
    number_of_FOVs: int = None
    nucChannel: list[int] = None
    cytoChannel: list[int] = None
    FISHChannel: list[int] = None
    voxel_size_z: int = 300  # This is voxel
    independent_params: dict = None
    kwargs: dict = None
    timestep_s: float = None

    def __init__(self, **kwargs):
        # Loop over all fields defined in the dataclass
        for f in fields(self):
            # Set the attribute with the value from kwargs, or the default if not provided
            setattr(self, f.name, kwargs.get(f.name, f.default))

    def __post_init__(self):

        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def pipeline_init(self):
        self.initial_data_location = pathlib.Path(self.initial_data_location)


@dataclass
class DataContainer(Parameters):
    local_data_folder: pathlib.Path = None
    local_mask_folder: pathlib.Path = None
    total_num_imgs: int = None  # this is the same as the experiment.number_of_images_to_process
    list_image_names: list[str] = None
    paths_to_images: list[pathlib.Path] = None
    list_images: list[np.ndarray] = None
    list_nuc_masks: list[np.ndarray] = None
    list_cell_masks: list[np.ndarray] = None
    list_cyto_mask: list[np.ndarray] = None
    num_img_2_run: int = None

    def pipeline_init(self):
        if self.num_img_2_run is None:
            self.num_img_2_run = self.total_num_imgs

        # if we dont have a local mask folder make one with the masks saved to it ================================
        if (self.local_mask_folder is None):
            self.save_masks_as_file = True
        else:
            self.save_masks_as_file = False

    def append(self, output):
        attributes = output.__dict__.keys()

        # if its a default pipeline data field
        for attr in attributes:
            if hasattr(self, attr):
                if getattr(output, attr) is not None and len(getattr(output, attr)) != 0:
                    setattr(self, attr, getattr(output, attr))
            else:
                setattr(self, attr, getattr(output, attr))
        
        # if it comes from a step
        if hasattr(self, output.__class__.__name__):
            getattr(self, output.__class__.__name__).append(output)
        else:
            setattr(self, output.__class__.__name__, output)


repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Settings(Parameters):
    return_data_to_NAS: int = 1
    NUMBER_OF_CORES: int = 4
    save_files: int = 1
    user_select_number_of_images_to_run: int = 100_000  # TODO: This is bad but I want it to select all and am too lazy
                                                        # to deal with it rn
    download_data_from_NAS: int = 0  # 0 for local, 1 for NAS
    connection_config_location: str = str(os.path.join(repo_path, 'config_nas.yml')) #r"C:\Users\Jack\Desktop\config_nas.yml" # r"/home/formanj/FISH_Processing_JF/FISH_Processing/config.yml"
    share_name: str = 'share'
    display_plots: bool = True
    load_in_mask: bool = False
    kwargs: dict = None

    def __init__(self, **kwargs):
        # Loop over all fields defined in the dataclass
        for f in fields(self):
            # Set the attribute with the value from kwargs, or the default if not provided
            setattr(self, f.name, kwargs.get(f.name, f.default))

    def __post_init__(self):
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)
