from dataclasses import dataclass


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
    NUMBER_Z_SLICES_TO_TRIM: int = None
    user_select_number_of_images_to_run: int = 100_000  # TODO: This is bad but I want it to select all and am too lazy
                                                        # to deal with it rn
    minimum_spots_cluster: int = 5
    show_plots: int = 1
    CLUSTER_RADIUS: float = 600  # int(psf_yx*1.5)
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
    MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: int = 3


        


