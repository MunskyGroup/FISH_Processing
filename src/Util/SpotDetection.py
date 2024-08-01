import numpy as np  # For numerical operations with arrays

# from . import Utilities, DataProcessing, BigFISH
from src.Util.BigFish import BigFISH
from src.Util.DataProcessing import DataProcessing
from src.Util.Utilities import Utilities


# from Util.BigFish import BigFISH             # For spot detection using Big-FISH
# from Utilities import Utilities         # Assuming a utility class for mask separation
# from Util.DataProcessing import DataProcessing  # Assuming a class for processing data


class SpotDetection():
    '''
    This class is intended to detect spots in FISH images using `Big-FISH <https://github.com/fish-quant/big-fish>`_. The format of the image must be  [Z, Y, X, C].
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter description obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
    For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    FISH_channels : int, or List
        List of channels with FISH spots that are used for the quantification
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    cluster_radius : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    masks_complete_cells : NumPy array
        Masks for every cell detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_nuclei: NumPy array
        Masks for every nucleus detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_cytosol_no_nuclei :  NumPy array
        Masks for every cytosol detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    dataframe : Pandas Dataframe 
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    list_voxels : List of tupples or None
        list with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each FISH channel.
        voxel_size_z is the height of a voxel, along the z axis, in nanometers. The default is 300.
        voxel_size_yx is the size of a voxel on the yx plan in nanometers. The default is 150.
    list_psfs : List of tuples or None
        List with a tuple with two elements (psf_z, psf_yx ) for each FISH channel.
        psf_z is the size of the PSF emitted by a spot in the z plan, in nanometers. The default is 350.
        psf_yx is the size of the PSF emitted by a spot in the yx plan in nanometers.
    show_plots : bool, optional
        If True, it shows a 2D maximum projection of the image and the detected spots. The default is False.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is False.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calculated automatically.
    
    '''
    def __init__(self,image,  FISH_channels ,channels_with_cytosol,channels_with_nucleus, cluster_radius=500, minimum_spots_cluster=4, masks_complete_cells = None, masks_nuclei  = None, masks_cytosol_no_nuclei = None, dataframe=None, image_counter=0, list_voxels=[[500,160]], list_psfs=[[350,160]], show_plots=True,image_name=None,save_all_images=True,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None,save_files=True):
        if len(image.shape)<4:
            image= np.expand_dims(image,axis =0)
        self.image = image
        self.number_color_channels = image.shape[-1]
        self.channels_with_cytosol=channels_with_cytosol
        self.channels_with_nucleus=channels_with_nucleus
        if not (masks_complete_cells is None):
            self.list_masks_complete_cells = Utilities().separate_masks(masks_complete_cells)
        elif (masks_complete_cells is None) and not(masks_nuclei is None):
            self.list_masks_complete_cells = Utilities().separate_masks(masks_nuclei)            
        if not (masks_nuclei is None):    
            self.list_masks_nuclei = Utilities().separate_masks(masks_nuclei)
        else:
            self.list_masks_nuclei = None
        
        if not (masks_complete_cells is None) and not (masks_nuclei is None):
            self.list_masks_cytosol_no_nuclei = Utilities().separate_masks(masks_cytosol_no_nuclei)
        else:
            self.list_masks_cytosol_no_nuclei = None
        self.FISH_channels = FISH_channels
        self.cluster_radius = cluster_radius
        self.minimum_spots_cluster = minimum_spots_cluster
        self.dataframe = dataframe
        self.image_counter = image_counter
        self.show_plots = show_plots
        if type(list_voxels[0]) != list:
            self.list_voxels = [list_voxels]
        else:
            self.list_voxels = list_voxels
        if type(list_psfs[0]) != list:
            self.list_psfs = [list_psfs]
        else:
            self.list_psfs = list_psfs
        # converting FISH channels to a list
        if not (type(FISH_channels) is list):
            self.list_FISH_channels = [FISH_channels]
        else:
            self.list_FISH_channels = FISH_channels
        self.image_name = image_name
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        if not isinstance(threshold_for_spot_detection, list):
            threshold_for_spot_detection=[threshold_for_spot_detection]
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.save_files = save_files
    def get_dataframe(self):
        list_fish_images = []
        list_thresholds_spot_detection = []
        for i in range(0,len(self.list_FISH_channels)):
            #print('Spot Detection for Channel :', str(self.list_FISH_channels[i]) )
            if (i ==0):
                dataframe_FISH = self.dataframe 
                reset_cell_counter = False
            voxel_size_z = self.list_voxels[i][0]
            voxel_size_yx = self.list_voxels[i][1]
            psf_z = self.list_psfs[i][0] 
            psf_yx = self.list_psfs[i][1]
            [spotDetectionCSV, clusterDetectionCSV], image_filtered, threshold = BigFISH(self.image, self.list_FISH_channels[i], voxel_size_z = voxel_size_z,voxel_size_yx = voxel_size_yx, psf_z = psf_z, psf_yx = psf_yx, 
                                                                                cluster_radius=self.cluster_radius,minimum_spots_cluster=self.minimum_spots_cluster, show_plots=self.show_plots,image_name=self.image_name,
                                                                                save_all_images=self.save_all_images,display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes,use_log_filter_for_spot_detection =self.use_log_filter_for_spot_detection,
                                                                                threshold_for_spot_detection=self.threshold_for_spot_detection[i],save_files=self.save_files).detect()
            list_thresholds_spot_detection.append(threshold)
            # converting the psf to pixles
            yx_spot_size_in_px = np.max((1,int(voxel_size_yx / psf_yx))).astype('int')
            
            dataframe_FISH = DataProcessing(spotDetectionCSV, clusterDetectionCSV, self.image, self.list_masks_complete_cells, self.list_masks_nuclei, self.list_masks_cytosol_no_nuclei, self.channels_with_cytosol,self.channels_with_nucleus,
                                            yx_spot_size_in_px=yx_spot_size_in_px, dataframe =dataframe_FISH,reset_cell_counter=reset_cell_counter,image_counter = self.image_counter ,spot_type=i,number_color_channels=self.number_color_channels ).get_dataframe()
            # reset counter for image and cell number
            #if i >0:
            reset_cell_counter = True
            list_fish_images.append(image_filtered)
        return dataframe_FISH, list_fish_images, list_thresholds_spot_detection

