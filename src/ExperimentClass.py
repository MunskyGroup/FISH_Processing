import pathlib
from dataclasses import dataclass, field


@dataclass
class Experiment:
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
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def pipeline_init(self):
        self.initial_data_location = pathlib.Path(self.initial_data_location)

    def to_dict(self):
        return self.__dict__


