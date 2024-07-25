import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns



class ColocalizationDistance():
    '''
    This class is intended to calculate the Euclidean 2nd norm distance between the spots detected in two FISH channels.
    
    Parameters
    
    dataframe : Pandas Dataframe 
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. 
        The default must contain spots detected in two different color channels.
    list_spot_type_to_compare : list, optional
        List indicating the combination of two values in spot_type to compare from the dataframe. The default is list_spot_type_to_compare =[0,1] indicating that spot_types 0 and 1 are compared.
    time_point : int, optional.
        Integer indicating the time point at which the data was collected. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_intensity_0 : int, optional
        Integer indicating the intensity threshold used to collected the data for the first color channel. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_intensity_1 : int, optional
        Integer indicating the intensity threshold used to collected the data for the second color channel. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_distance : float, optional.
        This number indicates the threshold distance in pixels that is used to determine if two spots are co-located in two different color channels if they are located inside this threshold_distance. The default value is 2.
    show_plots : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    voxel_size_z, voxel_size_yx: float, optional.
        These values indicate the microscope voxel size. These parameters are optional and should be included only if a normalization to the z-axis is needed to calculate distance.
    psf_z, psf_yx: float, optional.
        These values indicate the microscope point spread function value. These parameters are optional and should be included only if a normalization to the z-axis is needed to calculate distance.
    report_codetected_spots_in_both_channels : bool, optional
        This option report the number of co-detected spots in channel both channels. Notice that this represents the total number of codetected spots in ch0 and ch1. The default is True.
    '''
    def __init__(self, df,list_spot_type_to_compare =[0,1], time_point=0,threshold_intensity_0=0,threshold_intensity_1=0,threshold_distance=2,show_plots = False,voxel_size_z=None,psf_z=None,voxel_size_yx=None,psf_yx=None,report_codetected_spots_in_both_channels=False):
        self.df = df
        self.time_point= time_point
        self.threshold_intensity_0 = threshold_intensity_0
        self.threshold_intensity_1 = threshold_intensity_1
        self.threshold_distance = threshold_distance
        self.show_plots = show_plots
        self.list_spot_type_to_compare = list_spot_type_to_compare
        if not (voxel_size_z is None):
            self.scale = np.array ([ voxel_size_z/psf_z, voxel_size_yx/psf_yx, voxel_size_yx/psf_yx ])
        else:
            self.scale = 1
        self.report_codetected_spots_in_both_channels = report_codetected_spots_in_both_channels
    
    def extract_spot_classification_from_df(self):
        '''
        This method calculates the distance between the spots detected in two color channnels.
        
        Returns
        
        dataframe : Pandas dataframe
            Pandas dataframe with the following columns: [time, ts_intensity_0, ts_intensity_1, ts_distance, image_id, cell_id, num_0_only, num_1_only, num_0_1, num_0, num_1, total]. 
                num_0_only = num_type_0_only
                num_1_only = num_type_1_only
                num_0_1 = num_type_0_1
                num_0 = num_type_0_only + num_type_0_1
                num_1 = num_type_1_only + num_type_0_1
                num_0_total = total number of spots detected on ch 0.
                num_1_total = total number of spots detected on ch 1.
                total = num_type_0_only + num_type_1_only + num_type_0_1
                
        '''
        number_cells = self.df['cell_id'].nunique()
        array_spot_type_per_cell = np.zeros((number_cells, 14)).astype(int) # this array will store the spots separated  as types: spot_0_only, spot_1_only, or spot_0_1
        list_coordinates_colocalized_spots=[]
        list_coordinates_spots_0_only = []
        list_coordinates_spots_1_only = []
        for cell_id in range(number_cells):
            image_id = self.df[self.df["cell_id"] == cell_id]['image_id'].values[0]
            # retrieving the coordinates for spots type 0 and 1 for each cell 
            spot_type_0 = self.list_spot_type_to_compare[0] 
            spot_type_1 = self.list_spot_type_to_compare[1]
            array_spots_0 = np.asarray( self.df[['z','y','x']][(self.df["cell_id"] == cell_id) & (self.df["spot_type"] == spot_type_0)] ) # coordinates for spot_type_0 with shape [num_spots_type_0, 3]
            array_spots_1 = np.asarray( self.df[['z','y','x']][(self.df["cell_id"] == cell_id) & (self.df["spot_type"] == spot_type_1)] ) # coordinates for spot_type_1 with shape [num_spots_type_1, 3]
            total_spots0 = array_spots_0.shape[0]
            total_spots1 = array_spots_1.shape[0]            
            # Concatenating arrays from spots 0 and 1
            array_all_spots = np.concatenate((array_spots_0,array_spots_1), axis=0) 
            # Calculating a distance matrix. 
            distance_matrix = np.zeros( (array_all_spots.shape[0], array_all_spots.shape[0])) #  the distance matrix is an square matrix resulting from the concatenation of both spot  types.
            for i in range(len(array_all_spots)):
                for j in range(len(array_all_spots)):
                    if j<i:
                        distance_matrix[i,j] = np.linalg.norm( ( array_all_spots[i,:]-array_all_spots[j,:] ) * self.scale )
            # masking the distance matrix. Ones indicate the distance is less or equal than threshold_distance
            mask_distance_matrix = (distance_matrix <= self.threshold_distance) 
            # Selecting the right-lower quadrant as a subsection of the distance matrix that compares one spot type versus the other. 
            subsection_mask_distance_matrix = mask_distance_matrix[total_spots0:, 0:total_spots0].copy()
            index_true_distance_matrix = np.transpose((subsection_mask_distance_matrix==1).nonzero())
            # To calulate 0 and 1 spots only the negation (NOT) of the subsection_mask_distance_matrix is used.
            negation_subsection_mask_distance_matrix = ~subsection_mask_distance_matrix
            # creating a subdataframe containing the coordinates of colocalized spots
            colocalized_spots_in_spots0 = index_true_distance_matrix[:,1] # selecting the x-axis in [Y,X] matrix
            coordinates_colocalized_spots = array_spots_0[ colocalized_spots_in_spots0]
            #coordinates_colocalized_spots = array_spots_0[ index_true_distance_matrix[:,1]]
            column_with_cell_id = np.zeros((coordinates_colocalized_spots.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_colocalized_spots = np.hstack((coordinates_colocalized_spots, column_with_cell_id))   # append column
            list_coordinates_colocalized_spots.append(coordinates_colocalized_spots)
            # creating a subdataframe containing the coordinates of 0_only spots
            is_spot_only_type_0 = np.all(negation_subsection_mask_distance_matrix, axis =0 ) # Testing if all the columns are ones of inv(subsection_mask_distance_matrix). Representing spot type 0. Notice that np.all(arr, axis=0) does the calculation along the columns.
            localized_spots_in_spots_0_only = (is_spot_only_type_0 > 0).nonzero() #index_false_distance_matrix[:,1] # selecting the x-axis in [Y,X] matrix for 0_only spots
            coordinates_spots_0_only = array_spots_0[ localized_spots_in_spots_0_only]
            column_with_cell_id_0_only = np.zeros((coordinates_spots_0_only.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_0_only = np.hstack((coordinates_spots_0_only, column_with_cell_id_0_only))   # append column
            list_coordinates_spots_0_only.append(coordinates_spots_0_only)
            # creating a subdataframe containing the coordinates of 1_only spots
            is_spot_only_type_1 = np.all(negation_subsection_mask_distance_matrix, axis =1 ) #  Testing if all the rows are ones of inv(subsection_mask_distance_matrix). Representing spot type 1. Notice that np.all(arr, axis=1) does the calculation along the rows.    
            localized_spots_in_spots_1_only = (is_spot_only_type_1 > 0).nonzero() # index_false_distance_matrix[:,0] # selecting the y-axis in [Y,X] matrix for 1_only spots
            coordinates_spots_1_only = array_spots_1[ localized_spots_in_spots_1_only]
            column_with_cell_id_1_only = np.zeros((coordinates_spots_1_only.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_1_only = np.hstack((coordinates_spots_1_only, column_with_cell_id_1_only))   # append column
            list_coordinates_spots_1_only.append(coordinates_spots_1_only)
            # plotting the distance matrix. True values indicate that the combination of spots are inside the minimal selected radius.
            if self.show_plots == True:
                print('Cell_Id: ', str(cell_id))
                plt.imshow(subsection_mask_distance_matrix, cmap='Greys_r')
                plt.title('Subsection bool mask distance matrix') 
                plt.xlabel('Spots 0')
                plt.ylabel('Spots 1')   
                plt.show()        
            # Calculating each type of spots in cell
            num_type_0_only = coordinates_spots_0_only.shape[0]#np.sum(is_spot_only_type_0) 
            num_type_1_only = coordinates_spots_1_only.shape[0]#np.sum(is_spot_only_type_1) 
            #num_type_0_1 =  coordinates_colocalized_spots.shape[0] # This will display the number of colocalized spots only in channel 0
            if self.report_codetected_spots_in_both_channels == True:
                num_type_0_1 =  (total_spots0 - num_type_0_only) + (total_spots1 - num_type_1_only) # Number of spots in both channels
                total_spots = num_type_0_only+num_type_1_only+num_type_0_1
            else:
                num_type_0_1 =  coordinates_colocalized_spots.shape[0] # This will display the number of colocalized spots only in channel 0
                total_spots = num_type_0_only+num_type_1_only+num_type_0_1
            array_spot_type_per_cell[cell_id,:] = np.array([self.time_point, 
                                                            self.threshold_intensity_0, 
                                                            self.threshold_intensity_1, 
                                                            self.threshold_distance, 
                                                            image_id, 
                                                            cell_id, 
                                                            num_type_0_only, 
                                                            num_type_1_only, 
                                                            num_type_0_1, 
                                                            num_type_0_only+num_type_0_1, 
                                                            num_type_1_only+num_type_0_1, 
                                                            total_spots0,
                                                            total_spots1,
                                                            total_spots]).astype(int)
            list_labels = ['time','ts_intensity_0','ts_intensity_1','ts_distance','image_id','cell_id','num_0_only','num_1_only','num_0_1','num_0', 'num_1','num_0_total','num_1_total','total']
            # creating a dataframe
            df_spots_classification = pd.DataFrame(data=array_spot_type_per_cell, columns=list_labels)
            del coordinates_colocalized_spots,is_spot_only_type_0,is_spot_only_type_1,coordinates_spots_0_only,coordinates_spots_1_only
        # Creating dataframes for coordinates
        list_labels_coordinates = ['z','y','x','cell_id']
        new_dtypes = { 'cell_id':int, 'z':int,'y':int,'x':int}
        # Colocalized spots
        coordinates_colocalized_spots_all_cells = np.concatenate(list_coordinates_colocalized_spots)
        df_coordinates_colocalized_spots = pd.DataFrame(data=coordinates_colocalized_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_colocalized_spots = df_coordinates_colocalized_spots.astype(new_dtypes)
        # 0-only spots
        coordinates_0_only_spots_all_cells = np.concatenate(list_coordinates_spots_0_only)
        df_coordinates_0_only_spots = pd.DataFrame(data=coordinates_0_only_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_0_only_spots = df_coordinates_0_only_spots.astype(new_dtypes)
        # 1-only spots
        coordinates_1_only_spots_all_cells = np.concatenate(list_coordinates_spots_1_only)
        df_coordinates_1_only_spots = pd.DataFrame(data=coordinates_1_only_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_1_only_spots = df_coordinates_1_only_spots.astype(new_dtypes)
        return df_spots_classification, df_coordinates_colocalized_spots, df_coordinates_0_only_spots, df_coordinates_1_only_spots

