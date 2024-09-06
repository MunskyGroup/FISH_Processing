import pathlib
import numpy as np
from dataclasses import dataclass

@dataclass
class PipelineDataClass:
    local_data_folder: pathlib.Path = None
    local_mask_folder: pathlib.Path = None
    total_num_imgs: int = None  # this is the same as the experiment.number_of_images_to_process
    list_image_names: list[str] = None
    paths_to_images: list[pathlib.Path] = None
    list_images: list[np.ndarray] = None
    mask_nuclei: list[np.ndarray] = None
    mask_complete_cell: list[np.ndarray] = None
    mask_cytosol: list[np.ndarray] = None
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
        if hasattr(self, output.__class__.__name__):
            getattr(self, output.__class__.__name__).append(output)
        else:
            setattr(self, output.__class__.__name__, output)
