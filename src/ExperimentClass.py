import pathlib
from dataclasses import dataclass


@dataclass
class Experiment:
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image


    """
    initial_data_location: str
    number_of_images_to_process: int = None  # This will be all images to process and will be the product of the number of tp and number of FOVs
    number_of_channels: int = None
    number_of_timepoints: int = None
    number_of_Z: int = None
    number_of_FOVs: int = None
    nucChannel: list[int] = None
    cytoChannel: list[int] = None
    FISHChannel: list[int] = None
    voxel_size_z: int = 300  # This is voxel


    def __post_init__(self):
        self.initial_data_location = pathlib.Path(self.initial_data_location)


