import bigfish.multistack as multistack
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation

from src.Util.Intensity import Intensity
# from . import Utilities, Intensity
from src.Util.Utilities import Utilities


# from Utilities import Utilities
# from Util.Intensity import Intensity


class DataProcessing:
    '''
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter descriptions obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    spotDetectionCSV: np.int64 Array with shape (nb_clusters, 5) or (nb_clusters, 4). 
        One coordinate per dimension for the cluster\'s centroid (zyx or yx coordinates), the number of spots detected in the clusters, and its index.
    clusterDetectionCSV : np.int64 with shape (nb_spots, 4) or (nb_spots, 3).
        Coordinates of the detected spots . One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, the value is -1.
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    masks_complete_cells : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_nuclei: List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    channels_with_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_with_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    yx_spot_size_in_px : int
        Size of the FISH spot in pixels.
    spot_type : int, optional
        A label indicating the spot type, this counter starts at zero, increasing with the number of channels containing FISH spots. The default is zero.
    dataframe : Pandas dataframe or None.
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    reset_cell_counter : bool
        This number is used to reset the counter of the number of cells. The default is False.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    '''
    def __init__(self,spotDetectionCSV, clusterDetectionCSV, image, masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei,  channels_with_cytosol, channels_with_nucleus, yx_spot_size_in_px,  spot_type=0, dataframe =None,reset_cell_counter=False,image_counter=0,number_color_channels=None):
        self.spotDetectionCSV=spotDetectionCSV 
        self.clusterDetectionCSV=clusterDetectionCSV
        self.channels_with_cytosol=channels_with_cytosol
        self.channels_with_nucleus=channels_with_nucleus
        self.number_color_channels=number_color_channels
        self.yx_spot_size_in_px =yx_spot_size_in_px
        if len(image.shape)<4:
            image= np.expand_dims(image,axis =0)
        self.image = image
        if isinstance(masks_complete_cells, list) or (masks_complete_cells is None):
            self.masks_complete_cells=masks_complete_cells
        else:
            self.masks_complete_cells=Utilities().separate_masks(masks_complete_cells)
            
        if isinstance(masks_nuclei, list) or (masks_nuclei is None):
            self.masks_nuclei=masks_nuclei
        else:
            self.masks_nuclei=Utilities().separate_masks(masks_nuclei)  

        if isinstance(masks_cytosol_no_nuclei, list) or (masks_cytosol_no_nuclei is None):
            self.masks_cytosol_no_nuclei=masks_cytosol_no_nuclei
        else:
            self.masks_cytosol_no_nuclei= Utilities().separate_masks(masks_cytosol_no_nuclei)
        self.dataframe=dataframe
        self.spot_type = spot_type
        self.reset_cell_counter = reset_cell_counter
        self.image_counter = image_counter
        
        # This number represent the number of columns that doesnt change with the number of color channels in the image
        self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME = 18
    def get_dataframe(self):
        '''
        This method extracts data from the class SpotDetection and returns the data as a dataframe.
        
        Returns
        
        dataframe : Pandas dataframe
            Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented.
        '''
        def mask_selector(mask,calculate_centroid= True):
            mask_area = np.count_nonzero(mask)
            if calculate_centroid == True:
                centroid_y,centroid_x = ndimage.measurements.center_of_mass(mask)
            else:
                centroid_y,centroid_x = 0,0
            return  mask_area, int(centroid_y), int(centroid_x)
        def replace_border_px_with_zeros(mask,number_of_pixels_to_replace_in_border=3):
            mask[:number_of_pixels_to_replace_in_border, :] = 0
            mask[-number_of_pixels_to_replace_in_border:, :] = 0
            mask[:, :number_of_pixels_to_replace_in_border] = 0
            mask[:, -number_of_pixels_to_replace_in_border:] = 0
            return mask
        
        def testing_intenisty_calculation(temp_img,temp_masked_img,color_channel=0):
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            # Plot temp_img in the first subplot
            axs[0].imshow(temp_img)
            axs[0].set_title('temp_img ch' + str(color_channel))
            # Plot temp_masked_img in the second subplot
            axs[1].imshow(temp_masked_img)
            axs[1].set_title('temp_masked_img')
            axs[2].hist(temp_masked_img[np.nonzero(temp_masked_img)], bins=30)
            # Display the plot
            plt.show()
            print('color channel ',color_channel)
            print ('img ' ,temp_masked_img[np.nonzero(temp_masked_img)])
            print('len ',len (temp_masked_img[np.nonzero(temp_masked_img)]))
            print('mean',int( temp_masked_img[np.nonzero(temp_masked_img)].mean() ) )
            print('std ',int( temp_masked_img[np.nonzero(temp_masked_img)].std() ) )
            num_zeros = np.count_nonzero(temp_masked_img[np.nonzero(temp_masked_img)] == 0)
            print("Number of zeros: ", num_zeros )
            return None
        
        def data_to_df(df, spotDetectionCSV, clusterDetectionCSV, mask_nuc = None, mask_cytosol_only=None,masks_complete_cells=None, nuc_area = 0, cyto_area =0, cell_area=0,
                        nuc_centroid_y=0, nuc_centroid_x=0, cyto_centroid_y=0, cyto_centroid_x=0, image_counter=0, is_cell_in_border = 0, spot_type=0, cell_counter =0,
                        nuc_int=None, cyto_int = None, complete_cell_int=None,pseudo_cyto_int=None,nucleus_cytosol_intensity_ratio=None,nucleus_pseudo_cytosol_intensity_ratio=None):
            # spotDetectionCSV      nrna x  [Z,Y,X,idx_foci]
            # clusterDetectionCSV   nc   x  [Z,Y,X,size,idx_foci]
            # Removing TS from the image and calculating RNA in nucleus
            if not (self.channels_with_nucleus in (None,[None]) ):
                try:
                    # removing borders in mask
                    mask_nuc = replace_border_px_with_zeros(mask_nuc)
                    spots_no_ts, _, ts = multistack.remove_transcription_site(spotDetectionCSV, clusterDetectionCSV, mask_nuc, ndim=3)
                except:
                    spots_no_ts, ts = spotDetectionCSV, None
            else:
                spots_no_ts, ts = spotDetectionCSV, None
            #rna_out_ts      [Z,Y,X,idx_foci]         Coordinates of the detected RNAs with shape. One coordinate per dimension (zyx or yx coordinates) plus the index of the foci assigned to the RNA. If no foci was assigned, value is -1. RNAs from transcription sites are removed.
            #foci            [Z,Y,X,size, idx_foci]   One coordinate per dimension for the foci centroid (zyx or yx coordinates), the number of RNAs detected in the foci and its index.
            #ts              [Z,Y,X,size,idx_ts]      One coordinate per dimension for the transcription site centroid (zyx or yx coordinates), the number of RNAs detected in the transcription site and its index.
            if not (self.channels_with_nucleus in (None,[None]) ):
                try:
                    # removing borders in mask
                    mask_nuc = replace_border_px_with_zeros(mask_nuc)
                    spots_nuc, _ = multistack.identify_objects_in_region(mask_nuc, spots_no_ts, ndim=3)
                except:
                    spots_nuc = None
            else:
                spots_nuc = None
            # Detecting spots in the cytosol
            if not (self.channels_with_cytosol in (None,[None]) ) and not (self.channels_with_nucleus in (None,[None]) ):
                try:
                    # removing borders in mask 
                    mask_cytosol_only = replace_border_px_with_zeros(mask_cytosol_only)
                    spots_cytosol_only, _ = multistack.identify_objects_in_region(mask_cytosol_only, spotDetectionCSV[:,:3], ndim=3)
                except:
                    spots_cytosol_only = None
            elif not (self.channels_with_cytosol in (None,[None]))  and (self.channels_with_nucleus in (None,[None]) ):
                try:
                    # removing borders in mask
                    masks_complete_cells = replace_border_px_with_zeros(masks_complete_cells)
                    spots_cytosol_only, _ = multistack.identify_objects_in_region(masks_complete_cells, spotDetectionCSV[:,:3], ndim=3)
                except:
                    spots_cytosol_only = None
            else:
                spots_cytosol_only = None
            # coord_innp  Coordinates of the objects detected inside the region.
            # coord_outnp Coordinates of the objects detected outside the region.
            # ts                     n x [Z,Y,X,size,idx_ts]
            # spots_nuc              n x [Z,Y,X]
            # spots_cytosol_only     n x [Z,Y,X]
            number_columns = len(df.columns)
            if not(spots_nuc is None):
                num_ts = ts.shape[0]
                num_nuc = spots_nuc.shape[0]
            else:
                num_ts = 0
                num_nuc = 0
            if not(spots_cytosol_only is None) :
                num_cyto = spots_cytosol_only.shape[0] 
            else:
                num_cyto = 0
            # creating empty arrays if spots are detected in nucleus and cytosol
            if num_ts > 0:
                array_ts =                  np.zeros( ( num_ts,number_columns)  )
                spot_idx_ts =  np.arange(0,                  num_ts                                                 ,1 )
                detected_ts = True
            else:
                spot_idx_ts = []
                detected_ts = False
            if num_nuc > 0:
                array_spots_nuc =           np.zeros( ( num_nuc,number_columns) )
                spot_idx_nuc = np.arange(num_ts,             num_ts + num_nuc                                       ,1 )
                detected_nuc = True
            else:
                spot_idx_nuc = []
                detected_nuc = False
            if num_cyto>0:
                array_spots_cytosol_only =  np.zeros( ( num_cyto  ,number_columns) )
                spot_idx_cyt = np.arange(num_ts + num_nuc,   num_ts + num_nuc + num_cyto  ,1 )
                detected_cyto = True
            else:
                spot_idx_cyt = []
                detected_cyto = False
            # Spot index
            spot_idx = np.concatenate((spot_idx_ts,  spot_idx_nuc, spot_idx_cyt )).astype(int)
            
            # Populating arrays
            if not (self.channels_with_nucleus in (None,[None]) ):
                if detected_ts == True:
                    array_ts[:,10:13] = ts[:,:3]         # populating coord 
                    array_ts[:,13] = 1                  # is_nuc
                    array_ts[:,14] = 1                  # is_cluster
                    array_ts[:,15] =  ts[:,3]           # cluster_size
                    array_ts[:,16] = spot_type          # spot_type
                    array_ts[:,17] = is_cell_in_border  # is_cell_fragmented
                if detected_nuc == True:
                    array_spots_nuc[:,10:13] = spots_nuc[:,:3]   # populating coord 
                    array_spots_nuc[:,13] = 1                   # is_nuc
                    array_spots_nuc[:,14] = 0                   # is_cluster
                    array_spots_nuc[:,15] = 0                   # cluster_size
                    array_spots_nuc[:,16] =  spot_type          # spot_type
                    array_spots_nuc[:,17] =  is_cell_in_border  # is_cell_fragmented
            
            if not (self.channels_with_cytosol in (None,[None]) ) :
                if detected_cyto == True:
                    array_spots_cytosol_only[:,10:13] = spots_cytosol_only[:,:3]    # populating coord 
                    array_spots_cytosol_only[:,13] = 0                             # is_nuc
                    array_spots_cytosol_only[:,14] = 0                             # is_cluster
                    array_spots_cytosol_only[:,15] = 0                            # cluster_size
                    array_spots_cytosol_only[:,16] =  spot_type                    # spot_type
                    array_spots_cytosol_only[:,17] =  is_cell_in_border            # is_cell_fragmented
            # concatenate array
            if (detected_ts == True) and (detected_nuc == True) and (detected_cyto == True):
                array_complete = np.vstack((array_ts, array_spots_nuc, array_spots_cytosol_only))
            elif (detected_ts == True) and (detected_nuc == True) and (detected_cyto == False):
                array_complete = np.vstack((array_ts, array_spots_nuc))
            elif (detected_ts == False) and (detected_nuc == True) and (detected_cyto == False):
                array_complete =  array_spots_nuc
            elif(detected_ts == False) and (detected_nuc == True) and (detected_cyto == True):
                array_complete = np.vstack(( array_spots_nuc, array_spots_cytosol_only))
            elif(detected_ts == True) and (detected_nuc == False) and (detected_cyto == True):
                array_complete = np.vstack(( array_ts, array_spots_cytosol_only))
            elif(detected_ts == False) and (detected_nuc == False) and (detected_cyto == True):
                array_complete = array_spots_cytosol_only
            else:
                array_complete = np.zeros( ( 1,number_columns)  )
            # Saves a dataframe with zeros when no spots are detected on the cell.
            if array_complete.shape[0] ==1:
                # if NO spots are detected populate  with -1
                array_complete[:,2] = -1     # spot_id
                array_complete[:,8:self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME] = -1
                array_complete[:,13] = 0                             # is_nuc
                array_complete[:,14] = 0                             # is_cluster
                array_complete[:,15] = 0                             # cluster_size
                array_complete[:,16] = -1                           # spot_type
                array_complete[:,17] =  is_cell_in_border            # is_cell_fragmented
            else:
                # if spots are detected populate  the reported  array
                array_complete[:,2] = spot_idx.T     # spot_id
            # populating  array with cell  information
            array_complete[:,0] = image_counter  # image_id
            array_complete[:,1] = cell_counter   # cell_id
            
            array_complete[:,3] = nuc_centroid_y     #'nuc_y_centoid'
            array_complete[:,4] = nuc_centroid_x     #'nuc_x_centoid'
            
            array_complete[:,5] = cyto_centroid_y     #'cyto_y_centoid'
            array_complete[:,6] = cyto_centroid_x     #'cyto_x_centoid'
        
            array_complete[:,7] = nuc_area       #'nuc_area_px'
            array_complete[:,8] = cyto_area      # cyto_area_px
            array_complete[:,9] = cell_area      #'cell_area_px'
            
            # Populating array to add the average intensity in the cell
            for c in range (self.number_color_channels):
                if not (self.channels_with_nucleus in (None,[None]) ):
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+c] = nuc_int[c] 
                if not (self.channels_with_cytosol in (None,[None]) ) :
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels+c] = cyto_int[c]    
                if not (self.channels_with_cytosol in (None,[None]) ) :
                    # populating with complete_cell_int
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*2+c] = complete_cell_int[c]  
                # populating with pseudo_cyto_int
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*3+c] = pseudo_cyto_int[c]  
                # populating with nucleus_cytosol_intensity_ratio
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*4+c] = nucleus_cytosol_intensity_ratio[c]  
                # populating with nucleus_pseudo_cytosol_intensity_ratio
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*5+c] = nucleus_pseudo_cytosol_intensity_ratio[c]  

            # This section calculates the intenisty fo each spot and cluster
            # ts                     n x [Z,Y,X,size,idx_ts]
            # spots_nuc              n x [Z,Y,X]
            # spots_cytosol_only     n x [Z,Y,X]
            if num_ts >0:
                cluster_spot_size = (ts[:,3]*self.yx_spot_size_in_px).astype('int')
                intensity_ts = Intensity(original_image=self.image, spot_size=cluster_spot_size, array_spot_location_z_y_x=ts[:,0:3],  method = 'disk_donut').calculate_intensity()[0]
            if num_nuc >0:
                intensity_spots_nuc = Intensity(original_image=self.image, spot_size=self.yx_spot_size_in_px, array_spot_location_z_y_x=spots_nuc[:,0:3],  method = 'disk_donut').calculate_intensity()[0]
            if num_cyto >0 :
                intensity_spots_cyto = Intensity(original_image=self.image, spot_size=self.yx_spot_size_in_px, array_spot_location_z_y_x=spots_cytosol_only[:,0:3],  method = 'disk_donut').calculate_intensity()[0]
            
            # Cancatenating the final array
            if (detected_ts == True) and (detected_nuc == True) and (detected_cyto == True):
                array_spot_int = np.vstack((intensity_ts, intensity_spots_nuc, intensity_spots_cyto))
            elif (detected_ts == True) and (detected_nuc == True) and (detected_cyto == False):
                array_spot_int = np.vstack((intensity_ts, intensity_spots_nuc))
            elif (detected_ts == False) and (detected_nuc == True) and (detected_cyto == False):
                array_spot_int = intensity_spots_nuc
            elif(detected_ts == False) and (detected_nuc == True) and (detected_cyto == True):
                array_spot_int = np.vstack(( intensity_spots_nuc, intensity_spots_cyto))
            elif(detected_ts == True) and (detected_nuc == False) and (detected_cyto == True):
                array_spot_int = np.vstack(( intensity_ts, intensity_spots_cyto))
            elif(detected_ts == False) and (detected_nuc == False) and (detected_cyto == True):
                array_spot_int = intensity_spots_cyto
            else:
                array_spot_int = np.zeros( ( 1,self.number_color_channels )  )
            # Populating the columns wth the spots intensity for each channel.
            number_columns_after_adding_intensity_in_cell = self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+(6*self.number_color_channels )                       
            array_complete[:, number_columns_after_adding_intensity_in_cell: number_columns_after_adding_intensity_in_cell+self.number_color_channels] = array_spot_int
            # Creating the dataframe  
            #df = df.append(pd.DataFrame(array_complete, columns=df.columns), ignore_index=True)
            df_new = pd.DataFrame(array_complete, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)
            new_dtypes = {'image_id':int, 'cell_id':int, 'spot_id':int,'is_nuc':int,'is_cluster':int,'nuc_loc_y':int, 'nuc_loc_x':int,'cyto_loc_y':int, 'cyto_loc_x':int,'nuc_area_px':int,'cyto_area_px':int, 'cell_area_px':int,'x':int,'y':int,'z':int,'cluster_size':int,'spot_type':int,'is_cell_fragmented':int}
            df = df.astype(new_dtypes)
            return df
        
        if not (self.masks_nuclei is None):
            n_masks = len(self.masks_nuclei)
        else:
            n_masks = len(self.masks_complete_cells)  
            
        # Initializing Dataframe
        if (not ( self.dataframe is None))   and  ( self.reset_cell_counter == False): # IF the dataframe exist and not reset for multi-channel fish is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) +1
        
        elif (not ( self.dataframe is None)) and (self.reset_cell_counter == True):    # IF dataframe exist and reset is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) - n_masks +1   # restarting the counter for the number of cells
        
        elif self.dataframe is None: # IF the dataframe does not exist.
            # Generate columns for the number of color channels
            list_columns_intensity_nuc = []
            list_columns_intensity_cyto = []
            list_columns_intensity_complete_cell =[]
            list_intensity_spots = []
            list_intensity_clusters = []
            list_nucleus_cytosol_intensity_ratio =[]
            list_columns_intensity_pseudo_cyto=[]
            list_nucleus_pseudo_cytosol_intensity_ratio=[]
            for c in range(self.number_color_channels):
                list_columns_intensity_nuc.append( 'nuc_int_ch_' + str(c) )
                list_columns_intensity_cyto.append( 'cyto_int_ch_' + str(c) )
                list_columns_intensity_complete_cell.append( 'complete_cell_int_ch_' + str(c) )
                list_intensity_spots.append( 'spot_int_ch_' + str(c) )
                list_nucleus_cytosol_intensity_ratio.append('nuc_cyto_int_ratio_ch_' + str(c) )
                list_columns_intensity_pseudo_cyto.append('pseudo_cyto_int_ch_' + str(c) )
                list_nucleus_pseudo_cytosol_intensity_ratio.append('nuc_pseudo_cyto_int_ratio_ch_' + str(c) )
            # creating the main dataframe with column names
            new_dataframe = pd.DataFrame( columns= ['image_id', 'cell_id', 'spot_id','nuc_loc_y', 'nuc_loc_x','cyto_loc_y', 'cyto_loc_x','nuc_area_px','cyto_area_px', 'cell_area_px','z', 'y', 'x','is_nuc','is_cluster','cluster_size','spot_type','is_cell_fragmented'] + list_columns_intensity_nuc + list_columns_intensity_cyto +list_columns_intensity_complete_cell+list_columns_intensity_pseudo_cyto + list_nucleus_cytosol_intensity_ratio+list_nucleus_pseudo_cytosol_intensity_ratio +list_intensity_spots+list_intensity_clusters )
            counter_total_cells = 0
        # loop for each cell in image
        
        num_pixels_to_dilate = 30
        for id_cell in range (0,n_masks): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
            # calculating nuclear area and center of mass
            if not (self.channels_with_nucleus in  (None, [None])):
                nuc_area, nuc_centroid_y, nuc_centroid_x = mask_selector(self.masks_nuclei[id_cell], calculate_centroid=True)
                selected_mask_nuc = self.masks_nuclei[id_cell]
                dilated_image_mask = binary_dilation(selected_mask_nuc, iterations=num_pixels_to_dilate).astype('int')
                pseudo_cytosol_mask = np.subtract(dilated_image_mask, selected_mask_nuc)
                pseudo_cyto_int = np.zeros( (self.number_color_channels ))
                tested_mask_for_border =  self.masks_nuclei[id_cell]
                nuc_int = np.zeros( (self.number_color_channels ))
                for k in range(self.number_color_channels ):
                    temp_img = np.max (self.image[:,:,:,k ],axis=0)
                    temp_masked_img = temp_img * self.masks_nuclei[id_cell]
                    temp_masked_img_with_pseudo_cytosol_mask = temp_img * pseudo_cytosol_mask
                    nuc_int[k] =  np.round( temp_masked_img[np.nonzero(temp_masked_img)].mean() , 5)
                    pseudo_cyto_int[k] =  np.round( temp_masked_img_with_pseudo_cytosol_mask[np.nonzero(temp_masked_img_with_pseudo_cytosol_mask)].mean() , 5)
                    # if k ==0:
                    #     print('nucleus intensity calculation')
                    #     testing_intenisty_calculation(temp_img,temp_masked_img,color_channel=k)
                    #     print('pseudo_cytosol intensity calculation')
                    #     print('max',np.max(dilated_image_mask))
                    #     print('min,max',np.min(pseudo_cytosol_mask),np.max(pseudo_cytosol_mask), )
                    #     testing_intenisty_calculation(temp_img,temp_masked_img_with_pseudo_cytosol_mask,color_channel=k)
                    del temp_img, temp_masked_img,temp_masked_img_with_pseudo_cytosol_mask
            else:
                nuc_area, nuc_centroid_y, nuc_centroid_x = 0,0,0
                selected_mask_nuc = None
                nuc_int = None
                pseudo_cyto_int = np.zeros( (self.number_color_channels )) 
            # calculating cytosol area and center of mass
            if not (self.channels_with_cytosol in (None, [None])):
                cell_area, cyto_centroid_y, cyto_centroid_x  = mask_selector(self.masks_complete_cells[id_cell],calculate_centroid=True)
                tested_mask_for_border =  self.masks_complete_cells[id_cell]
                complete_cell_int = np.zeros( (self.number_color_channels ))
                cyto_int = np.zeros( (self.number_color_channels ))
                for k in range(self.number_color_channels ):
                    temp_img = np.max (self.image[:,:,:,k ],axis=0)
                    # calculating cytosol intensity for complete cell mask
                    temp_masked_img = temp_img * self.masks_complete_cells[id_cell]
                    complete_cell_int[k] =  np.round( temp_masked_img[np.nonzero(temp_masked_img)].mean() , 5) 
                    # calculating cytosol intensity only. Removing the nucleus
                    temp_masked_img_cyto_only = temp_img * self.masks_cytosol_no_nuclei[id_cell]
                    cyto_int[k]=  np.round( temp_masked_img_cyto_only[np.nonzero(temp_masked_img_cyto_only)].mean() , 5)
                    #if k ==0:
                    #    print('complete cell intensity calculation')
                    #    testing_intenisty_calculation(temp_img,temp_masked_img,color_channel=k)
                    #    print('cytosol only intensity calculation')
                    #    testing_intenisty_calculation(temp_img,temp_masked_img_cyto_only,color_channel=k)
                    del temp_img, temp_masked_img, temp_masked_img_cyto_only
            else:
                complete_cell_int = None
                cell_area, cyto_centroid_y, cyto_centroid_x  = 0,0,0
                cyto_int = None
            
            # Calculating ratio between nucleus and cytosol intensity
            nucleus_cytosol_intensity_ratio = np.zeros( (self.number_color_channels ))
            nucleus_pseudo_cytosol_intensity_ratio = np.zeros( (self.number_color_channels ))
            # case where nucleus and cyto are passed 
            if not (self.channels_with_cytosol in (None, [None])) and not (self.channels_with_nucleus in  (None, [None])):
                for k in range(self.number_color_channels ):
                    nucleus_cytosol_intensity_ratio[k] = nuc_int[k]/ cyto_int[k]
                    nucleus_pseudo_cytosol_intensity_ratio[k] = nuc_int[k]/ pseudo_cyto_int[k]
            # case where nucleus is  passed but not cyto
            elif (self.channels_with_cytosol in (None, [None])) and not (self.channels_with_nucleus in  (None, [None])):
                for k in range(self.number_color_channels ):
                    nucleus_pseudo_cytosol_intensity_ratio[k] = nuc_int[k]/ pseudo_cyto_int[k]
            # case where nucleus and cyto are passed 
            if not (self.channels_with_cytosol in (None, [None])) and not (self.channels_with_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei = self.masks_cytosol_no_nuclei[id_cell]
                cyto_area,_,_ = mask_selector(self.masks_cytosol_no_nuclei[id_cell],calculate_centroid=False)
                selected_masks_complete_cells = self.masks_complete_cells[id_cell]
            # case where nucleus is  passed but not cyto
            elif (self.channels_with_cytosol in (None, [None])) and not (self.channels_with_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei = None
                cyto_area = 0
                selected_masks_complete_cells = None
            # case where cyto is passed but not nucleus
            elif not (self.channels_with_cytosol in (None, [None])) and (self.channels_with_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei,_,_ = mask_selector( self.masks_complete_cells[id_cell],calculate_centroid=False) 
                cyto_area, _, _  = mask_selector(self.masks_complete_cells[id_cell],calculate_centroid=False) # if not nucleus channel is passed the cytosol is consider the complete cell.
                selected_masks_complete_cells = self.masks_complete_cells[id_cell]
            else:
                slected_masks_cytosol_no_nuclei = None
                cyto_area = 0 
                selected_masks_complete_cells = None
            # determining if the cell is in the border of the image. If true the cell is in the border.
            is_cell_in_border =  np.any( np.concatenate( ( tested_mask_for_border[:,0],tested_mask_for_border[:,-1],tested_mask_for_border[0,:],tested_mask_for_border[-1,:] ) ) )  
            # Data extraction
            new_dataframe = data_to_df( new_dataframe, 
                                        self.spotDetectionCSV, 
                                        self.clusterDetectionCSV, 
                                        mask_nuc = selected_mask_nuc, 
                                        mask_cytosol_only=slected_masks_cytosol_no_nuclei, 
                                        masks_complete_cells = selected_masks_complete_cells,
                                        nuc_area=nuc_area,
                                        cyto_area=cyto_area, 
                                        cell_area=cell_area, 
                                        nuc_centroid_y = nuc_centroid_y, 
                                        nuc_centroid_x = nuc_centroid_x,
                                        cyto_centroid_y = cyto_centroid_y, 
                                        cyto_centroid_x = cyto_centroid_x,
                                        image_counter=self.image_counter,
                                        is_cell_in_border = is_cell_in_border,
                                        spot_type = self.spot_type ,
                                        cell_counter =counter_total_cells,
                                        nuc_int=nuc_int,
                                        cyto_int=cyto_int,
                                        complete_cell_int = complete_cell_int,
                                        pseudo_cyto_int=pseudo_cyto_int,
                                        nucleus_cytosol_intensity_ratio=nucleus_cytosol_intensity_ratio,
                                        nucleus_pseudo_cytosol_intensity_ratio=nucleus_pseudo_cytosol_intensity_ratio)
            counter_total_cells +=1
        return new_dataframe

