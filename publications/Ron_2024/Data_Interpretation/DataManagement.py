# This class is intended to process the data from the CSV file and save the result as a new CSV file that summarizes the results for a given experiment.

# Importing modules
import os
import pandas as pd
import numpy as np
import pathlib
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
current_dir = pathlib.Path().absolute()
fa_dir = current_dir.parents[2].joinpath('src')
# Importing fish_analyses module
sys.path.append(str(fa_dir))
import fish_analyses as fa


class DataManagement:
    def __init__(self, file_path, Condition, DexConc, Replica, time_value,time_TPL_value,output_path,minimum_spots_cluster=2,mandatory_substring=None,connect_to_NAS=False,path_to_config_file=None,save_csv=True):
        # This section downloads the zip file from NAS and extracts the dataframe or uses the dataframe from the local folder.
        if connect_to_NAS == False:
            self.file_path = file_path
        else:
            # download zip file from NAS
            local_folder_path = pathlib.Path().absolute().joinpath('temp_zip_analyses')
            local_folder_path.mkdir(exist_ok=True)
            share_name = 'share'
            # Reading the data from NAS, unziping files, organizing data as single dataframe for comparison. 
            list_local_files = fa.Utilities().read_zipfiles_from_NAS([file_path],path_to_config_file,share_name, mandatory_substring, local_folder_path)
            list_local_folders = fa.Utilities().unzip_local_folders(list_local_files,local_folder_path)
            self.file_path = list_local_folders[0]
            # Extracting the dataframe
            self.file_path =  glob.glob( str(list_local_folders[0].joinpath('dataframe_*')) )[0]
        # Defining the attributes
        self.Condition = Condition
        self.DexConc = DexConc
        self.Replica = Replica
        self.time_value = time_value
        self.time_TPL_value = time_TPL_value
        self.output_path=output_path
        self.minimum_spots_cluster=minimum_spots_cluster
        self.save_csv=save_csv
    
    def data_processor(self):
        """
        Processes the data from the CSV file and saves the result as a new CSV file that summarizes the results for a given experiment.

        Returns:
            pandas.DataFrame: The processed DataFrame with fields 
            cell_id, Condition, Replica, Dex_Conc, Time_index, Time_TPL, Nuc_Area, Cyto_Area, Nuc_GR_avg_int, Cyto_GR_avg_int, Nuc_DUSP1_avg_int, Cyto_DUSP1_avg_int, RNA_DUSP1_nuc, RNA_DUSP1_cyto, DUSP1_ts_size_0, DUSP1_ts_size_1, DUSP1_ts_size_2, DUSP1_ts_size_3.
        """
        dataframe = pd.read_csv(self.file_path)
        # Defining constant values for all cells
        number_cells = dataframe['cell_id'].nunique()
        time_index = self.time_value 
        dex_conc_index = self.DexConc 
        TPL_time_index = self.time_TPL_value 
        
        # Initialize empty lists to store values for each cell
        nuc_area_list = []  
        cyto_area_list = []  
        GR_avg_nuc_intensity_list = []  
        GR_avg_cyto_intensity_list = []  
        DUSP1_avg_nuc_intensity_list = []
        DUSP1_avg_cyto_intensity_list = []
        RNA_nuc_list = []  
        RNA_cyto_list = []  
        ts_size_list = []
        
        for i in range(number_cells):
            nuc_area = np.asarray(dataframe.loc[
                (dataframe['cell_id'] == i)
            ].nuc_area_px.values[0])
            
            if self.Condition == 'DUSP1_timesweep' or self.Condition == 'DUSP1_TPL':
                cyto_area = np.asarray(dataframe.loc[
                    (dataframe['cell_id'] == i)
                ].cyto_area_px.values[0])
            elif self.Condition == 'GR_timesweep': # This is the case when the cytosol is not segmented.
                cyto_area = np.nan
            
            nuc_int = np.asarray(dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_cell_fragmented'] != -1)
            ].nuc_int_ch_0.values[0])
            
            if self.Condition == 'DUSP1_timesweep' or self.Condition == 'DUSP1_TPL':
                cyto_int = np.asarray(dataframe.loc[
                    (dataframe['cell_id'] == i) &
                    (dataframe['is_cell_fragmented'] != -1)
                ].cyto_int_ch_0.values[0])
            elif self.Condition == 'GR_timesweep': 
                cyto_int = np.asarray(dataframe.loc[
                    (dataframe['cell_id'] == i) &
                    (dataframe['is_cell_fragmented'] != -1)
                ].pseudo_cyto_int_ch_0.values[0])
            
            # Count the number of RNA in the nucleus
            nuc_spots = len(dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_nuc'] == True) &
                (dataframe['is_cell_fragmented'] != -1)
            ].spot_id)
            
            nuc_cluster_rna = dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_nuc'] == True) &
                (dataframe['is_cell_fragmented'] != -1)
            ].cluster_size.sum()
            
            nuc = np.asarray(nuc_spots + nuc_cluster_rna - 1)
            
            # Count the number of RNA in the cytoplasm
            cyto_spots = len(dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_nuc'] == False) &
                (dataframe['is_cell_fragmented'] != -1)
            ].spot_id)
            
            cyto_cluster_rna = dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_nuc'] == False) &
                (dataframe['is_cell_fragmented'] != -1)
            ].cluster_size.sum()
            
            cyto = np.asarray(cyto_spots + cyto_cluster_rna - 1)
            
            ####### This is counting all transcription sites for DUSP1 that are larger than "minimum_spots_cluster".
            ts_size = dataframe.loc[
                (dataframe['cell_id'] == i) &
                (dataframe['is_cluster'] == True) &
                (dataframe['is_nuc'] == True) &
                (dataframe['cluster_size'] >= self.minimum_spots_cluster) &
                (dataframe['is_cell_fragmented'] != -1)
            ].cluster_size.values
            ts_size = np.sort(ts_size)[::-1]# Sort ts_size values in descending order
            ts_size_array = np.asarray(ts_size[:4])  # Select the first 4 values
            
            # appending the values to the lists
            nuc_area_list.append(nuc_area)
            cyto_area_list.append(cyto_area)
            
            if self.Condition == 'GR_timesweep':
                DUSP1_avg_nuc_intensity_list.append(np.nan)
                DUSP1_avg_cyto_intensity_list.append(np.nan)
                GR_avg_nuc_intensity_list.append(nuc_int)
                GR_avg_cyto_intensity_list.append(cyto_int)
                
            elif self.Condition == 'DUSP1_timesweep' or self.Condition == 'DUSP1_TPL':
                DUSP1_avg_nuc_intensity_list.append(nuc_int)
                DUSP1_avg_cyto_intensity_list.append(cyto_int)
                GR_avg_nuc_intensity_list.append(np.nan)
                GR_avg_cyto_intensity_list.append(np.nan)
            
            RNA_nuc_list.append(nuc)
            RNA_cyto_list.append(cyto)
            ts_size_list.append(ts_size_array)

        
        # Create a pandas DataFrame from the list of ts_int values
        df_ts_size_per_cell = pd.DataFrame(ts_size_list) 
        df_ts_size_per_cell.columns = [f"DUSP1_ts_size_{i}" for i in range(len(df_ts_size_per_cell.columns))] # 

        # Create a dictionary with the data
        data = {
            'Cell_id': np.arange(number_cells),  
            'Condition': [self.Condition] * number_cells,  
            'Replica': [self.Replica] * number_cells,  
            'Dex_Conc': [dex_conc_index] * number_cells,  
            'Time_index': [time_index] * number_cells,  
            'Time_TPL': TPL_time_index, # Time that triptolide is added relative to Dex application.
            'Nuc_Area': nuc_area_list, 
            'Cyto_Area': cyto_area_list, # Only relevant for condition == DUSP1_timesweep and DUSP1_TPL. NaNs for GR_timesweep.
            'Nuc_GR_avg_int': GR_avg_nuc_intensity_list,
            'Cyto_GR_avg_int': GR_avg_cyto_intensity_list,
            'Nuc_DUSP1_avg_int': DUSP1_avg_nuc_intensity_list, # Only relevant for condition == DUSP1_timesweep and DUSP1_TPL. NaNs for GR_timesweep.
            'Cyto_DUSP1_avg_int': DUSP1_avg_cyto_intensity_list, # Only relevant for condition == DUSP1_timesweep and DUSP1_TPL. NaNs for GR_timesweep.
            'RNA_DUSP1_nuc': RNA_nuc_list,    # RNA_GR_cyto do we need also for DUSP1?
            'RNA_DUSP1_cyto': RNA_cyto_list  # RNA_GR_cyto do we need also for DUSP1?
        }
        # Create a pandas DataFrame from the dictionary
        df_data = pd.DataFrame(data)
        
        # Check the condition and set NaNs accordingly
        if self.Condition == 'GR_timesweep':
            df_data['Time_TPL'] = np.nan
            df_data['RNA_DUSP1_nuc'] = np.nan
            df_data['RNA_DUSP1_cyto'] = np.nan
            for i in range(4):  # Assuming there are always 3 TS columns
                df_ts_size_per_cell[f"DUSP1_ts_size_{i}"] = np.nan
        elif self.Condition == 'DUSP1_timesweep':
            df_data['Time_TPL'] = np.nan                
            df_data['Nuc_GR_avg_int'] = np.nan
            df_data['Cyto_GR_avg_int'] = np.nan
        elif self.Condition == 'DUSP1_TPL':
            df_data['Nuc_GR_avg_int'] = np.nan
            df_data['Cyto_GR_avg_int'] = np.nan    
        
        # Concatenate df_data and df_ts_size_per_cell column-wise
        complete_df = pd.concat([df_data, df_ts_size_per_cell], axis=1)
        # Define the file path and name for saving the result
        result_file_path = os.path.join(self.output_path, os.path.basename(self.file_path).replace('.csv', '_result.csv'))
        
        # Save the processed DataFrame as a CSV file
        if self.save_csv == True:
            complete_df.to_csv(result_file_path, index=False, na_rep='NaN')

        return complete_df


