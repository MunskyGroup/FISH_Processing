import pathlib

class Experiment():
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image


    """
    def __init__(self, 
                initial_data_location: str,
                number_of_images_to_process: int = None,  # This will be all images to process and will be the product of the number of tp and number of FOVs
                number_of_channels: int = None,
                number_of_timepoints: int = None,
                number_of_Z: int = None,
                number_of_FOVs: int = None,
                nucChannel: list[int] = None,
                cytoChannel: list[int] = None,
                FISHChannel: list[int] = None,
                voxel_size_z: int = 500,):
        self.initial_data_location = pathlib.Path(initial_data_location)
        self.number_of_images_to_process = number_of_images_to_process
        self.number_of_channels = number_of_channels
        self.number_of_timepoints = number_of_timepoints
        self.nucChannel = nucChannel
        self.cytoChannel = cytoChannel
        self.FISHChannel = FISHChannel
        self.number_of_Z = number_of_Z
        self.number_of_FOVs = number_of_FOVs
        self.voxel_size_z = voxel_size_z


