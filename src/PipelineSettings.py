from dataclasses import dataclass, fields


@dataclass
class PipelineSettings:
    return_data_to_NAS: int = 1
    optimization_segmentation_method: str = 'default'
    save_all_imgs: int = 1
    save_all_filtered_imgs: int = 0
    threshold_for_spot_detection: str = None
    NUMBER_OF_CORES: int = 4
    save_filtered_images: int = 1
    remove_z_slices_borders: int = 1
    remove_out_of_focus_images: int = 1
    save_pdf_report: int = 1
    MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE: int = 5
    NUMBER_Z_SLICES_TO_TRIM: int = 0
    user_select_number_of_images_to_run: int = 100_000  # TODO: This is bad but I want it to select all and am too lazy
                                                        # to deal with it rn
    minimum_spots_cluster: int = 5
    show_plots: int = 1
    CLUSTER_RADIUS: float = 750  # int(psf_yx*1.5)
    save_all_images: int = 1
    display_spots_on_multiple_z_planes: int = 0
    use_log_filter_for_spot_detection: int = 0
    sharpness_threshold: float = 1.10
    local_or_NAS: int = 0  # 0 for local, 1 for NAS
    save_files: int = 1
    model_nuc_segmentation: str = None
    model_cyto_segmentation: str = None
    pretrained_model_nuc_segmentation: str = None
    pretrained_model_cyto_segmentation: str = None
    diameter_nucleus: float = 100
    diameter_cytosol: float = 200
    save_masks_as_file: int = 0
    MINIMAL_NUMBER_OF_PIXELS_IN_MASK: int = 1000
    MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: int = 20
    MINUMUM_NUMBERs_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: int = 3
    connection_config_location: str = r"C:\Users\Jack\Desktop\config.yml" # r"/home/formanj/FISH_Processing_JF/FISH_Processing/config.yml"
    share_name: str = 'share'
    display_plots: bool = True
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

    def pipeline_init(self):
        pass

    def to_dict(self):
        return self.__dict__
    


        


