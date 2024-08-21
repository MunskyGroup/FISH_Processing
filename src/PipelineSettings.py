class PipelineSettings():
    def __init__(self, 
                 return_data_to_NAS:int=1,
                 optimization_segmentation_method: str = 'default',
                 save_all_imgs: int = 1,
                 save_all_filtered_imgs: int = 0,
                 threshold_for_spot_detection: str = None,
                 NUMBER_OF_CORES: int = 4,
                 save_filtered_images: int = 1,
                 remove_z_slices_borders: int = 1,
                 remove_out_of_focus_images: int = 1,
                 save_pdf_report: int = 1,
                 MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE:int = 5, 
                 NUMBER_Z_SLICES_TO_TRIM: int = None,
                 User_Select_number_of_Images_to_run: int = 1000,
                 minimum_spots_cluster: int = 5,
                 show_plots: int = 1,
                 CLUSTER_RADIUS: float = 600, #int(psf_yx*1.5)
                 save_all_images: int = 1,
                 display_spots_on_multiple_z_planes: int = 0,
                 use_log_filter_for_spot_detection: int = 0,
                 sharpness_threshold: float = 1.10,
                 local_or_NAS: int = 0, # 0 for local, 1 for NAS
                 save_files: int = 1,
                 model_nuc_segmentation: str = None,
                 model_cyto_segmentation: str = None,
                 pretrained_model_nuc_segmentation: str = None,
                 pretrained_model_cyto_segmentation: str = None,
                 diameter_nucleus: float = 100,
                 diameter_cytosol: float = 200,
                 save_masks_as_file: int = 0,
                 MINIMAL_NUMBER_OF_PIXELS_IN_MASK: int = 1000,
                 MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: int = 20,
                 MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: int = 3,
                 ) -> None:
        
        self.user_select_number_of_images_to_run = User_Select_number_of_Images_to_run
        self.MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE = MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE
        self.return_data_to_NAS = return_data_to_NAS
        self.optimization_segmentation_method = optimization_segmentation_method
        self.save_all_imgs = save_all_imgs
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.save_filtered_images = save_filtered_images
        self.remove_z_slices_borders = remove_z_slices_borders
        self.remove_out_of_focus_images = remove_out_of_focus_images
        self.save_pdf_report = save_pdf_report
        self.save_all_filtered_imgs = save_all_filtered_imgs
        self.minimum_spots_cluster = minimum_spots_cluster
        self.show_plots = show_plots
        self.CLUSTER_RADIUS = CLUSTER_RADIUS 
        self.optimization_segmentation_method = optimization_segmentation_method # optimization_segmentation_method = 'default', 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection = use_log_filter_for_spot_detection
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.sharpness_threshold = sharpness_threshold
        self.local_or_NAS = local_or_NAS
        self.save_files = save_files
        self.model_nuc_segmentation = model_nuc_segmentation
        self.model_cyto_segmentation = model_cyto_segmentation
        self.pretrained_model_nuc_segmentation = pretrained_model_nuc_segmentation
        self.pretrained_model_cyto_segmentation = pretrained_model_cyto_segmentation
        self.diameter_nucleus = diameter_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.save_masks_as_file = save_masks_as_file
        self.MINIMAL_NUMBER_OF_PIXELS_IN_MASK = MINIMAL_NUMBER_OF_PIXELS_IN_MASK


        self.save_pdf_report = save_pdf_report
        self.save_files = save_files
        self.model_nuc_segmentation=model_nuc_segmentation
        self.model_cyto_segmentation=model_cyto_segmentation
        self.pretrained_model_nuc_segmentation=pretrained_model_nuc_segmentation
        self.pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation



        # flag to display spots on trimming z-planes
        self.NUMBER_Z_SLICES_TO_TRIM = NUMBER_Z_SLICES_TO_TRIM
        self.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = (
            MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD)
        self.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD
        


