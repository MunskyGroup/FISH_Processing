import pathlib
import numpy as np
from dataclasses import dataclass

@dataclass
class DataContainer:
    local_data_folder: pathlib.Path = None
    local_mask_folder: pathlib.Path = None
    total_num_imgs: int = None  # this is the same as the experiment.number_of_images_to_process
    list_image_names: list[str] = []
    paths_to_images: list[pathlib.Path] = None
    list_images: list[np.ndarray] = []
    list_nuc_masks: list[np.ndarray] = []
    list_cell_masks: list[np.ndarray] = []
    list_cyto_mask: list[np.ndarray] = []
    num_img_2_run: int = None

    def pipeline_init(self):
        if self.num_img_2_run is None:
            self.num_img_2_run = self.total_num_imgs

        # if we dont have a local mask folder make one with the masks saved to it ================================
        if (self.local_mask_folder is None):
            self.save_masks_as_file = True
        else:
            self.save_masks_as_file = False

    def _check_params(self):
        pass

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
