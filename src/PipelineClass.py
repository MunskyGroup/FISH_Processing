


class PipelineSettings():
    def __init__(self, return_data_to_NAS:int=1,
                 optimization_segmentation_method:str='default',
                 save_all_imgs:int=1,
                 threshold_for_spot_detection=None,
                 NUMBER_OF_CORES:int=4,
                 save_filtered_images:int=1,
                 remove_z_slices_borders:int=1,
                 remove_out_of_focus_images:int=1,
                 save_pdf_report:int=1):
        self.return_data_to_NAS = return_data_to_NAS
        self.optimization_segmentation_method = optimization_segmentation_method
        self.save_all_imgs = save_all_imgs
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.save_filtered_images = save_filtered_images
        self.remove_z_slices_borders = remove_z_slices_borders
        self.remove_out_of_focus_images = remove_out_of_focus_images
        self.save_pdf_report = save_pdf_report

    def get_return_data_to_NAS(self) -> int:
        return self.return_data_to_NAS
    
    def get_optimization_segmentation_method(self) -> str:
        return self.optimization_segmentation_method
    
    def get_save_all_imgs(self) -> int:
        return self.save_all_imgs
    
    def get_threshold_for_spot_detection(self):
        return self.threshold_for_spot_detection
    
    def get_NUMBER_OF_CORES(self) -> int:
        return self.NUMBER_OF_CORES
    
    def get_save_filtered_images(self) -> int:
        return self.save_filtered_images
    
    def get_remove_z_slices_borders(self) -> int:
        return self.remove_z_slices_borders
    
    def get_remove_out_of_focus_images(self) -> int:
        return self.remove_out_of_focus_images
    
    def get_save_pdf_report(self) -> int:
        return self.save_pdf_report
    
    def get_pipeline_settings(self) -> dict:
        return {'return_data_to_NAS': self.return_data_to_NAS,
                'optimization_segmentation_method': self.optimization_segmentation_method,
                'save_all_imgs': self.save_all_imgs,
                'threshold_for_spot_detection': self.threshold_for_spot_detection,
                'NUMBER_OF_CORES': self.NUMBER_OF_CORES,
                'save_filtered_images': self.save_filtered_images,
                'remove_z_slices_borders': self.remove_z_slices_borders,
                'remove_out_of_focus_images': self.remove_out_of_focus_images,
                'save_pdf_report': self.save_pdf_report}

