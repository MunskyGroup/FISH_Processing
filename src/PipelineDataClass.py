import pathlib
import numpy as np

class PipelineDataClass():
    def __init__(self,
                 local_data_folder: pathlib.Path = None,
                 local_mask_folder: pathlib.Path = None,
                 total_num_imgs: int = None,  # this is the same as the experiment.number_of_images_to_process
                 list_image_names: list[str] = None,
                 paths_to_images: list[pathlib.Path] = None,
                 list_images: list[np.ndarray] = None,
                 mask_nuclei: list[np.ndarray] = None,
                 mask_complete_cell: list[np.ndarray] = None,
                 mask_cytosol: list[np.ndarray] = None,
                 num_img_2_run: int = None,
                 ) -> None:
        self.local_data_folder = local_data_folder
        self.total_num_imgs = total_num_imgs
        self.list_image_names = list_image_names
        self.paths_to_images = paths_to_images
        self.list_images = list_images
        self.masks_nuclei = mask_nuclei
        self.masks_cytosol = mask_cytosol
        self.masks_complete_cells = mask_complete_cell

        if num_img_2_run is None:
            self.num_img_2_run = self.total_num_imgs


        self.local_mask_folder = local_mask_folder

        # if we dont have a local mask folder make one with the masks saved to it ================================
        if (local_mask_folder is None):
            self.save_masks_as_file = True
        else:
            self.save_masks_as_file = False


        '''Properties none made by initialization but by steps:
        self.pipelineData.list_z_slices_per_image = [img.shape[0] for img in list_images] ---- Z trimming
        '''


    def _check_params(self):
        pass

    def append(self, output):
        if hasattr(self, output.__class__.__name__):
            getattr(self, output.__class__.__name__).append(output)
        else:
            setattr(self, output.__class__.__name__, output)
