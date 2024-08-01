import itertools
import math
import os
import pathlib
import re
import warnings

import joypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal
from skimage.io import imread
from skimage.measure import find_contours
from skimage.morphology import erosion

font_props = {'size': 16}

# from . import Utilities, RemoveExtrema, GaussianFilter
from src.Util.Utilities import Utilities
from src.Util.RemoveExtrema import RemoveExtrema
from src.Util.GaussianFilter import GaussianFilter
# from Util.GaussianFilter import GaussianFilter
# from Utilities import Utilities
# from RemoveExtrema import RemoveExtrema

class Plots():
    '''
    This class contains miscellaneous methods to generate plots. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass
    
    def plotting_segmentation_images(self,directory,list_files_names,list_segmentation_successful=[None],image_name='temp.pdf',show_plots=True):
        number_images = len(list_files_names)
        NUM_COLUMNS = 1
        NUM_ROWS = number_images
        # Plotting
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (15, NUM_ROWS*3))
        # Prealocating plots
        for i in range (0, NUM_ROWS):
            if NUM_ROWS == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
        # Plotting each image
        for i in range(0, number_images):
            if NUM_ROWS == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            if (list_segmentation_successful[i] == True): # or (list_segmentation_successful[i] is None):
                temp_segmented_img_name = directory.joinpath('seg_' + list_files_names[i].split(".")[0] +'.png' )
                temp_img =  imread(str( temp_segmented_img_name ))
                axis_index.imshow( temp_img)
            img_title= list_files_names[i]
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title(img_title[:-4], fontsize=6 )
        plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plots ==True:
            plt.show()
        else:
            plt.close()
        plt.show()
        
        
    def plotting_all_original_images(self,list_images,list_files_names,image_name,show_plots=True):
        number_images = len(list_images)
        NUM_COLUMNS = 5
        NUM_ROWS = math.ceil(number_images/ NUM_COLUMNS)
        # Plotting
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (15, NUM_ROWS*3))
        # Prealocating plots
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        # Plotting each image
        r = 0
        c = 0
        counter = 0
        for i in range(0, number_images):
            if NUM_ROWS == 1:
                axis_index = axes[c]
            else:
                axis_index = axes[r,c]
            temp_img =  list_images[i] #imread(str( local_data_dir.joinpath(list_files_names[i]) ))
            max_image = np.max (temp_img,axis =0)
            max_nun_channels = np.min([3, max_image.shape[2]])
            img_title= list_files_names[i]
            image_int8 = Utilities().convert_to_int8(max_image[ :, :, 0:max_nun_channels], rescale=True, min_percentile=1, max_percentile=95)  
            axis_index.imshow( image_int8)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title(img_title[:-4], fontsize=6 )
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=1
            counter +=1
        plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plots ==True:
            plt.show()
        else:
            plt.close()
        plt.show()
        
        
        
    
    def plot_images(self,image,figsize=(8.5, 5),image_name='temp',show_plots=True, use_maximum_projection=False):
        '''
        This method is intended to plot all the channels from an image with format  [Z, Y, X, C].
        
        Parameters
        
        image: NumPy array
            Array of images with dimensions [Z, Y, X, C].
        figsize : tuple with figure size, optional.
            Tuple with format (x_size, y_size). the default is (8.5, 5).
        '''
        number_channels = image.shape[3]
        number_z_slices = image.shape[0]
        if number_z_slices ==1:
            center_slice =0
        else:
            center_slice = image.shape[0]//2
            
        _, axes = plt.subplots(nrows=1, ncols=number_channels, figsize=figsize)
        for i in range (0,number_channels ):
            if number_z_slices >1:
                if use_maximum_projection == True:
                    temp_max = np.max(image[:,:,:,i],axis =0)
                    rescaled_image = RemoveExtrema(temp_max,min_percentile=1, max_percentile=98).remove_outliers() 
                else:
                    rescaled_image = RemoveExtrema(image[center_slice,:,:,i],min_percentile=1, max_percentile=98).remove_outliers() 
            else:
                rescaled_image = RemoveExtrema(image[center_slice,:,:,i],min_percentile=1, max_percentile=98).remove_outliers() #image
            if number_channels ==1:
                axis_index = axes
                axis_index.imshow( rescaled_image ,cmap='Spectral') 
                #axis_index.imshow( rescaled_image[center_slice,:,:,i] ,cmap='Spectral') 
            else:
                axis_index = axes[i]
                axis_index.imshow( rescaled_image ,cmap='Spectral') 
            axis_index.set_title('Channel_'+str(i))
            axis_index.grid(color='k', ls = '-.', lw = 0.5)
        plt.savefig(image_name,bbox_inches='tight',dpi=180)

            
        if show_plots ==True:
            plt.show()
        else:
            plt.close()
        return None

    
    def plotting_masks_and_original_image(self,image, masks_complete_cells, masks_nuclei, channels_with_cytosol, channels_with_nucleus,image_name,show_plots,df_labels=None):
    # This functions makes zeros the border of the mask, it is used only for plotting.
        NUM_POINTS_MASK_EDGE_LINE = 50
        def erode_mask(img,px_to_remove = 1):
            img[0:px_to_remove, :] = 0;img[:, 0:px_to_remove] = 0;img[img.shape[0]-px_to_remove:img.shape[0]-1, :] = 0; img[:, img.shape[1]-px_to_remove: img.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
            return erosion(img) # performin erosion of the mask to remove not connected pixeles.
        # This section converst the image into a 2d maximum projection.
        if len(image.shape) > 3:  # [ZYXC]
            if image.shape[0] ==1:
                max_image = image[0,:,:,:]
            else:
                max_image = np.max(image[:,:,:,:],axis=0)    # taking the mean value
        else:
            max_image = image # [YXC] 
        # Plotting
        n_channels = np.min([3, max_image.shape[2]])
        im = Utilities().convert_to_int8(max_image[ :, :, 0:n_channels], rescale=True, min_percentile=1, max_percentile=95)  
        if np.max(masks_complete_cells) != 0 and not(channels_with_cytosol in (None,[None])) and not(channels_with_nucleus in (None,[None])):
            _, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 10))
            masks_plot_cyto= masks_complete_cells 
            masks_plot_nuc = masks_nuclei              
            axes[0].imshow(im)
            axes[0].set(title = 'All channels')
            axes[1].imshow(masks_plot_cyto)
            axes[1].set(title = 'Cytosol mask')
            axes[2].imshow(masks_plot_nuc)
            axes[2].set(title = 'Nuclei mask')
            axes[3].imshow(im)
            n_masks =np.max(masks_complete_cells)                 
            for i in range(1, n_masks+1 ):
                # Removing the borders just for plotting
                tested_mask_cyto = np.where(masks_complete_cells == i, 1, 0).astype(bool)
                tested_mask_nuc = np.where(masks_nuclei == i, 1, 0).astype(bool)
                # Remove border for plotting
                temp_nucleus_mask= erode_mask(tested_mask_nuc)
                temp_complete_mask = erode_mask(tested_mask_cyto)
                temp_nucleus_mask[0, :] = 0; temp_nucleus_mask[-1, :] = 0; temp_nucleus_mask[:, 0] = 0; temp_nucleus_mask[:, -1] = 0
                temp_complete_mask[0, :] = 0; temp_complete_mask[-1, :] = 0; temp_complete_mask[:, 0] = 0; temp_complete_mask[:, -1] = 0
                temp_contour_n = find_contours(temp_nucleus_mask, 0.1, fully_connected='high',positive_orientation='high')
                temp_contour_c = find_contours(temp_complete_mask, 0.1, fully_connected='high',positive_orientation='high')
                contours_connected_n = np.vstack((temp_contour_n))
                contour_n = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                if contour_n.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour_n = signal.resample(contour_n, num = NUM_POINTS_MASK_EDGE_LINE)
                contours_connected_c = np.vstack((temp_contour_c))
                contour_c = np.vstack((contours_connected_c[-1,:],contours_connected_c))
                if contour_c.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour_c = signal.resample(contour_c, num = NUM_POINTS_MASK_EDGE_LINE)
                axes[3].fill(contour_n[:, 1], contour_n[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask nucleus
                axes[3].fill(contour_c[:, 1], contour_c[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask cytosol
                axes[3].set(title = 'Paired masks')
            if not (df_labels is None):
                cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                for _, label in enumerate(cell_ids_labels):
                    cell_idx_string = str(label)
                    Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                    X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                    axes[3].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='black')
        else:
            if not(channels_with_cytosol in (None,[None])) and (channels_with_nucleus in (None,[None])):
                masks_plot_cyto= masks_complete_cells 
                n_channels = np.min([3, max_image.shape[2]])
                _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_cyto)
                axes[1].set(title = 'Cytosol mask')
                axes[2].imshow(im)
                n_masks =np.max(masks_complete_cells)                 
                for i in range(1, n_masks+1 ):
                    # Removing the borders just for plotting
                    tested_mask_cyto = np.where(masks_complete_cells == i, 1, 0).astype(bool)
                    # Remove border for plotting
                    temp_complete_mask = erode_mask(tested_mask_cyto)
                    temp_complete_mask[0, :] = 0; temp_complete_mask[-1, :] = 0; temp_complete_mask[:, 0] = 0; temp_complete_mask[:, -1] = 0
                    temp_contour_c = find_contours(temp_complete_mask, 0.1, fully_connected='high',positive_orientation='high')
                    try:
                        contours_connected_c = np.vstack((temp_contour_c))
                        contour_c = np.vstack((contours_connected_c[-1,:],contours_connected_c))
                        if contour_c.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                            contour_c = signal.resample(contour_c, num = NUM_POINTS_MASK_EDGE_LINE)
                            axes[2].fill(contour_c[:, 1], contour_c[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask cytosol
                    except:
                        contour_c = 0
                    axes[2].set(title = 'Original + Masks')
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'cyto_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'cyto_loc_x'].item()
                        axes[2].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='black')
            if (channels_with_cytosol in (None,[None])) and not(channels_with_nucleus in (None,[None])):
                masks_plot_nuc = masks_nuclei    
                _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_nuc)
                axes[1].set(title = 'Nuclei mask')
                axes[2].imshow(im)
                n_masks =np.max(masks_nuclei)                 
                for i in range(1, n_masks+1 ):
                    # Removing the borders just for plotting
                    tested_mask_nuc = np.where(masks_nuclei == i, 1, 0).astype(bool)
                    # Remove border for plotting
                    temp_nucleus_mask= erode_mask(tested_mask_nuc)
                    temp_nucleus_mask[0, :] = 0; temp_nucleus_mask[-1, :] = 0; temp_nucleus_mask[:, 0] = 0; temp_nucleus_mask[:, -1] = 0
                    temp_contour_n = find_contours(temp_nucleus_mask, 0.1, fully_connected='high',positive_orientation='high')
                    contours_connected_n = np.vstack((temp_contour_n))
                    contour_n = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                    if contour_n.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                        contour_n = signal.resample(contour_n, num = NUM_POINTS_MASK_EDGE_LINE)
                    axes[2].fill(contour_n[:, 1], contour_n[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask nucleus
                    axes[2].set(title = 'Original + Masks')
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                        axes[2].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='black')
        if not(image_name is None):
            plt.savefig(image_name,bbox_inches='tight',dpi=180)
        if show_plots == 1:
            plt.show()
        else:
            plt.close()


    def dist_plots(self, df, plot_title,destination_folder,y_lim_values=None ):
        stacked_df = df.stack()
        pct = stacked_df.quantile(q=0.99)
        #color_palete = 'colorblind'
        color_palete = 'CMRmap'
        #color_palete = 'OrRd'
        sns.set_style("white")
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        max_x_val = df.max().max()
        # Distribution
        plt.figure(figsize=(10,5))
        sns.set(font_scale = 1)
        sns.set_style("white")
        p_dist =sns.kdeplot(data=df,palette=color_palete,cut=0,lw=5)
        p_dist.set_xlabel("Spots")
        p_dist.set_ylabel("Kernel Density Estimator (KDE)")
        p_dist.set_title(plot_title)
        p_dist.set(xlim=(0, pct))
        name_plot = 'Dist_'+plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))

        # ECDF
        plt.figure(figsize=(10,5))
        sns.set(font_scale = 1)
        sns.set_style("white")
        p_dist =sns.ecdfplot(data=df,palette=color_palete,lw=5)
        p_dist.set_xlabel("Spots")
        p_dist.set_ylabel("Proportion")
        p_dist.set_title(plot_title)
        p_dist.set_ylim(0,1.05)
        p_dist.set(xlim=(0, pct))
        name_plot = 'ECDF_'+ plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))

        # Whisker Plots
        plt.figure(figsize=(7,9))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        p = sns.stripplot(data=df, size=3, color='0.5', jitter=0.2)
        plt.xticks(rotation=45, ha="right")
        sns.set(font_scale = 1.5)
        bp=sns.boxplot( 
                    meanprops={'visible': True,'color': 'r', 'ls': 'solid', 'lw': 4},
                    whiskerprops={'visible': True, 'color':'k','ls': 'solid', 'lw': 1},
                    data=df,
                    showcaps={'visible': False, 'color':'orangered', 'ls': 'solid', 'lw': 1}, # Q1-Q3 25-75%
                    ax=p,
                    showmeans=True,meanline=True,zorder=10,showfliers=False,showbox=True,linewidth=1,color='w')
        p.set_xlabel("Time After Treatment")
        p.set_ylabel("Spot Count")
        p.set_title(plot_title)
        if not (y_lim_values is None):
            p.set(ylim=y_lim_values)
        sns.set(font_scale = 1.5)
        name_plot = 'Whisker_'+plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        
        # Joy plots
        plt.figure(figsize=(7,5))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        fig, axes = joypy.joyplot(df,x_range=[-5,pct],bins=25,hist=False, overlap=0.8, linewidth=1, figsize=(7,5), colormap=cm.CMRmap) #
        name_plot = 'JoyPlot_'+ plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        
        return None

    def plot_comparing_df(self, df_all,df_cyto,df_nuc,plot_title,destination_folder):
        #color_palete = 'CMRmap'
        color_palete = 'Dark2'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        # This code creates a single colum for all conditions and adds a 'location' column.
        df_all_melt = df_all.melt()
        df_all_melt['location'] = 'all' 
        df_cyto_melt = df_cyto.melt()
        df_cyto_melt['location']= 'cyto'
        df_nuc_melt = df_nuc.melt()
        df_nuc_melt['location']= 'nuc' 
        data_frames_list = [df_all_melt, df_cyto_melt, df_nuc_melt]
        data_frames = pd.concat(data_frames_list)       
        # Plotting
        plt.figure(figsize=(12,7))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        b= sns.barplot(data=data_frames, x= 'variable',y='value', hue = 'location',palette=color_palete)
        b.set_xlabel("time after treatment")
        b.set_ylabel("Spot Count")
        b.set_title(plot_title)
        plt.xticks(rotation=45, ha="right") 
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plot_TS(self, df_original,plot_title,destination_folder,minimum_spots_cluster,remove_zeros=False):
        color_palete = 'CMRmap'
        #color_palete = 'Accent'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        df= df_original.copy()
        if remove_zeros == True:
            for col in df.columns:
                df[col] = np.where(df[col]==0, np.nan, df[col])
        plt.figure(figsize=(12,7))
        b= sns.stripplot(data=df, size=4, jitter=0.3, dodge=True,palette=color_palete)
        b.set_xlabel('time after treatment')
        b.set_ylabel('No. Cells with TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>)' )
        b.set_title(plot_title)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.xticks(rotation=45, ha="right")
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plot_TS_bar_stacked(self, df_original,plot_title,destination_folder,minimum_spots_cluster,remove_zeros=False,normalize=True):
        if (normalize == True) and (remove_zeros == True):
            warnings.warn("Warining: notice that normalization is only possible if zeros are not removed. To normalize the output use the options as follows: remove_zeros=False, normalize=True ")
        df= df_original.copy()
        color_palete = 'OrRd'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        if remove_zeros == True:
            for col in df.columns:
                df[col] = np.where(df[col]==0, np.nan, df[col])
            min_range = 1
            num_labels =4
            column_labels =['1 TS','2 TS','>2 TS']
            ts_values = list(range(min_range,num_labels )) # 1,2,3
            max_ts_count = 1
        else:
            min_range = 0
            num_labels =4
            column_labels =['0 TS','1 TS','2 TS','>2 TS']
            max_ts_count = 2
            ts_values = list(range(min_range, num_labels )) # 0,1,2,3
        num_columns = len(list(df.columns))
        table_data = np.zeros((len(ts_values),num_columns)) 
        
        for i, col in enumerate(df.columns):
            for indx, ts_size in enumerate (ts_values):
                if indx<=max_ts_count:
                    table_data[indx,i] = df.loc[df[col] == ts_size, col].count()
                else:
                    
                    table_data[indx,i] = df.loc[df[col] >= ts_size, col].count()
        
        if (normalize == True) and (remove_zeros == False):
            number_cells = np.sum(table_data,axis =0)
            normalized_table = table_data/number_cells
            df_new = pd.DataFrame(normalized_table.T, columns = column_labels,index=list(df.columns))
            ylabel_text = ' TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>) / Cell' 
        else:
            df_new = pd.DataFrame(table_data.T, columns = column_labels,index=list(df.columns))
            ylabel_text = 'No. Cells with TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>)' 
        # Plotting
        b= df_new.plot(kind='bar', stacked=True,figsize=(12,7)) #, cmap=color_palete
        b.legend(fontsize=12)
        b.set_xlabel('time after treatment')
        b.set_ylabel(ylabel_text )
        b.set_title(plot_title)
        if normalize == False:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.xticks(rotation=45, ha="right")
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plotting_results_as_distributions(self, number_of_spots_per_cell,number_of_spots_per_cell_cytosol,number_of_spots_per_cell_nucleus,ts_size,number_of_TS_per_cell,minimum_spots_cluster,numBins=20,output_identification_string=None, spot_type=0):
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Plotting intensity distributions
        def plot_probability_distribution(data_to_plot, numBins = 10, title='', xlab='', ylab='', color='r', subplots=False, show_grid=True, fig=plt.figure() ):
            n, bins, _ = plt.hist(data_to_plot,bins=numBins,density=False,color=color)
            plt.xlabel(xlab, size=16)
            plt.ylabel(ylab, size=16)
            plt.grid(show_grid)
            plt.text(bins[(len(bins)//2)],(np.amax(n)//2).astype(int),'mean = '+str(round( np.mean(data_to_plot) ,1) ), fontsize=14,bbox=dict(facecolor='w', alpha=0.5) )
            plt.title(title, size=16)
            return (f)
        # Section that generates each subplot
        number_subplots = int(np.any(number_of_spots_per_cell)) + int(np.any(number_of_spots_per_cell_cytosol)) + int(np.any(number_of_spots_per_cell_nucleus)) + int(np.any(ts_size)) + int(np.any(number_of_TS_per_cell))
        file_name = 'spot_distributions_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        #Plotting
        fig_size = (25, 5)
        f = plt.figure(figsize=fig_size)
        #ylab='Probability'
        ylab='Frequency Count' 
        selected_color = '#1C00FE' 
        # adding subplots
        subplot_counter = 0
        if np.any(number_of_spots_per_cell):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter) 
            plot_probability_distribution( number_of_spots_per_cell, numBins=20,  title='Total Num Spots per cell', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_spots_per_cell_cytosol):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_spots_per_cell_cytosol,   numBins=20,  title='Num Spots in Cytosol', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_spots_per_cell_nucleus):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_spots_per_cell_nucleus, numBins=20,    title='Num Spots in Nucleus', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(ts_size):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(ts_size, numBins=20,    title='Clusters in nucleus', xlab='RNA per Cluster', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_TS_per_cell):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_TS_per_cell ,  numBins=20, title='Number TS per cell', xlab='[TS (>= '+str(minimum_spots_cluster)+' rna)]', ylab=ylab, fig=f, color=selected_color)
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf' )
        plt.show()
        return file_name



    def plot_scatter_and_distributions(self, x,y, plot_title,  x_label_scatter='cell_size', y_lable_scatter = 'number_of_spots_per_cell', destination_folder=None, selected_color = '#1C00FE',save_plot=False,temporal_figure=False):
        r, p = stats.pearsonr(x, y)
        df_join_distribution = pd.DataFrame({x_label_scatter:x,y_lable_scatter:y})
        sns.set(font_scale = 1.3)
        b = sns.jointplot(data=df_join_distribution, y=y_lable_scatter, x=x_label_scatter, color= selected_color , marginal_kws=dict(bins=40, rug=True))
        b.plot_joint(sns.rugplot, height=0, color=[0.7,0.7,0.7], clip_on=True)
        b.plot_joint(sns.kdeplot, color=[0.5,0.5,0.5], levels=5)
        b.plot_joint(sns.regplot,scatter_kws={'color': 'orangered',"s":10, 'marker':'o'}, line_kws={'color': selected_color,'lw': 2} )
        blank_plot, = b.ax_joint.plot([], [], linestyle="", alpha=0)
        b.ax_joint.legend([blank_plot],['r={:.2f}'.format( np.round(r,2))],loc='upper left',)
        b.ax_joint.set_xlim(np.percentile(x,0.01), np.percentile(x,99.9))
        b.ax_joint.set_ylim(np.percentile(y,0.01), np.percentile(y,99.9))
        b.fig.suptitle(plot_title)
        b.ax_joint.collections[0].set_alpha(0)
        b.fig.tight_layout()
        b.fig.subplots_adjust(top=0.92) 
        name_plot = plot_title 
        if temporal_figure == True:
            file_name = 'temp__'+str(np.random.randint(1000, size=1)[0])+'__'+name_plot+'.png' # generating a random name for the temporal plot
            plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
            plt.close(b.fig)
        else:
            file_name = name_plot+'.pdf'
        if (save_plot == True) and (temporal_figure == False):
            plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
            plt.show()
        if not (destination_folder is None) and (save_plot == True) and (temporal_figure==False):
            pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,file_name))
        return b.fig, file_name

    def plot_cell_size_spots(self, channels_with_cytosol, channels_with_nucleus, cell_size, number_of_spots_per_cell, cyto_size, number_of_spots_per_cell_cytosol, nuc_size, number_of_spots_per_cell_nucleus,output_identification_string=None,spot_type=0):  
        '''
        This function is intended to plot the spot count as a function of the cell size. 
        
        '''
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        
        if not channels_with_cytosol in (None, 'None', 'none',['None'],['none'],[None]):
            cyto_exists = True
        else:
            cyto_exists = False
        if not channels_with_nucleus in (None, 'None', 'none',['None'],['none'],[None]):
            nuc_exists = True
        else:
            nuc_exists = False
        # Plot title
        title_plot='cell'
        file_name = 'scatter_cell_size_vs_spots_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        # Complete cell
        if (cyto_exists == True) and (nuc_exists == True):
            x = cell_size
            y = number_of_spots_per_cell
            _,fig1_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='cell_size', y_lable_scatter = 'number_of_spots_per_cell',temporal_figure=True)
        # Cytosol
        if cyto_exists == True:
            x = cyto_size
            y = number_of_spots_per_cell_cytosol
            title_plot='cytosol'
            _,fig2_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot ,x_label_scatter='cyto_size', y_lable_scatter = 'number_of_spots_per_cyto',temporal_figure=True)
        # Nucleus
        if nuc_exists == True:
            x = nuc_size
            y = number_of_spots_per_cell_nucleus
            title_plot='nucleus'
            _,fig3_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot ,x_label_scatter='nuc_size', y_lable_scatter = 'number_of_spots_per_nuc',temporal_figure=True)
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
        counter = 0
        if cyto_exists == True:
            axes[counter].imshow(plt.imread(fig2_temp_name))
            os.remove(fig2_temp_name)
            counter +=1
        if nuc_exists == True:
            axes[counter].imshow(plt.imread(fig3_temp_name))
            os.remove(fig3_temp_name)
        if (cyto_exists == True) and (nuc_exists == True):
            axes[2].imshow(plt.imread(fig1_temp_name))
            os.remove(fig1_temp_name)
        # removing axis
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].grid(False)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name

    def plot_cell_intensity_spots(self, dataframe, number_of_spots_per_cell_nucleus = None, number_of_spots_per_cell_cytosol = None,output_identification_string=None,spot_type=0):  
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Detecting if the nucleus and cytosol are detected
        if np.any(number_of_spots_per_cell_cytosol):
            cyto_exists = True
        else:
            cyto_exists = False
        if np.any(number_of_spots_per_cell_nucleus):
            nucleus_exists = True
        else:
            nucleus_exists = False
        if (nucleus_exists==True) and (cyto_exists==True):
            number_rows = 2
        else:
            number_rows = 1
        # Creating plot
        file_name  = 'ch_int_vs_spots_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        counter = 0
        _, axes = plt.subplots(nrows = number_rows, ncols = number_color_channels, figsize = (15, 10))
        for j in range(number_rows):
            for i in range(number_color_channels):
                if number_rows==1 and (number_color_channels==1):
                    axis_index = axes
                elif number_rows==1 and (number_color_channels>=1):
                    axis_index = axes[j]
                elif number_rows==2 and (number_color_channels>1):
                    axis_index = axes[j,i]                
                if (nucleus_exists==True) and (counter ==0):
                    column_with_intensity = 'nuc_int_ch_'+str(i)
                    title_plot='nucleus'
                    x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract=column_with_intensity, extraction_type='values_per_cell') 
                    y = number_of_spots_per_cell_nucleus
                if ((cyto_exists==True) and (counter ==1)) or ((cyto_exists==True) and (counter ==0) and (number_rows==1)):
                    column_with_intensity = 'cyto_int_ch_'+str(i)
                    title_plot='cytosol'
                    x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract=column_with_intensity, extraction_type='values_per_cell') 
                    y = number_of_spots_per_cell_cytosol
                _,fig_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='Intensity_Ch_'+str(i), y_lable_scatter = 'number_of_spots',temporal_figure=True)
                
                if number_rows ==1:
                    axis_index.imshow(plt.imread(fig_temp_name))
                    axis_index.grid(False)
                    axis_index.set_xticks([])
                    axis_index.set_yticks([])
                else:
                    axes[j,i].imshow(plt.imread(fig_temp_name))
                    axes[j,i].grid(False)
                    axes[j,i].set_xticks([])
                    axes[j,i].set_yticks([])
                os.remove(fig_temp_name)
                del x, y
            counter +=1
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name

    def plot_scatter_bleed_thru (self, dataframe,channels_with_cytosol, channels_with_nucleus,output_identification_string=None):
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Calculating the number of combination of color channels
        combinations_channels = list(itertools.combinations(range(number_color_channels), 2))
        _, axes = plt.subplots(nrows = 1, ncols = len(combinations_channels), figsize = (20, 10))
        for i in range(len(combinations_channels)):
            if len(combinations_channels) == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            title_plot=title_string
            file_name  = 'bleed_thru_'+title_string+'.pdf'
            if not channels_with_cytosol in (None, 'None', 'none',['None'],['none'],[None]):
                x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='cyto_int_ch_'+str(combinations_channels[i][0]), extraction_type='values_per_cell') 
                y = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='cyto_int_ch_'+str(combinations_channels[i][1]), extraction_type='values_per_cell') 
            if not channels_with_nucleus in (None, 'None', 'none',['None'],['none'],[None]):
                x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='nuc_int_ch_'+str(combinations_channels[i][0]), extraction_type='values_per_cell') 
                y = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='nuc_int_ch_'+str(combinations_channels[i][1]), extraction_type='values_per_cell') 
            _,fig_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='intensity_Ch_'+str(combinations_channels[i][0]), y_lable_scatter = 'intensity_Ch_'+str(combinations_channels[i][1]),temporal_figure=True)
            axis_index.imshow(plt.imread(fig_temp_name))
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            del x, y 
            os.remove(fig_temp_name)
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
    
    def plot_interpretation_distributions (self, df_all, df_cyto, df_nuc, destination_folder, plot_title_suffix='',y_lim_values_all_spots=None, y_lim_values_cyto=None,y_lim_values_nuc=None):
        if (df_cyto.dropna().any().any() == True) and (df_nuc.dropna().any().any() == True):  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            plot_title_complete = 'all_spots__'+plot_title_suffix
            Plots().dist_plots(df_all, plot_title_complete, destination_folder,y_lim_values_all_spots)
        
        if df_cyto.dropna().any().any() == True:  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            # Plotting for all Cytosol only
            plot_title_cyto = 'cyto__'+plot_title_suffix
            Plots().dist_plots(df_cyto, plot_title_cyto, destination_folder,y_lim_values_cyto)
        
        if df_nuc.dropna().any().any() == True:  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            # Plotting for all nucleus
            plot_title_nuc = 'nuc__'+plot_title_suffix
            Plots().dist_plots(df_nuc, plot_title_nuc, destination_folder,y_lim_values_nuc)


    def plot_spot_intensity_distributions(self, dataframe,output_identification_string=None,remove_outliers=True, spot_type=0):
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string=''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 5))
        max_percentile = 99
        min_percentile = 0.5
        title_plot  = 'spot_intensities'
        file_name = title_plot +'_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        colors = ['r','g','b','m']
        for i in range (0,number_color_channels ):
            if number_color_channels ==1:
                axis_index = axes
            else:
                axis_index = axes[i]
            column_name = 'spot_int_ch_'+str(i)
            df_spot_intensity = dataframe.loc[   (dataframe['is_cluster']==False) & (dataframe['spot_type']==spot_type)]
            spot_intensity = df_spot_intensity[column_name].values
            if remove_outliers ==True:
                spot_intensity =Utilities().remove_outliers( spot_intensity,min_percentile=1,max_percentile=98)
            axis_index.hist(x=spot_intensity, bins=30, density = True, histtype ='bar',color = colors[i],label = 'spots')
            axis_index.set_xlabel('spot intensity Ch_'+str(i) )
            axis_index.set_ylabel('probability' )
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
    
    
    def plot_nuc_cyto_int_ratio_distributions(self, dataframe,output_identification_string=None, plot_for_pseudo_cytosol=False,remove_outliers=True):
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string=''
        # Counting the number of color channels in the dataframe
        if plot_for_pseudo_cytosol == True:
            pattern = r'nuc_pseudo_cyto_int_ratio_ch_\d'
            title_plot  = 'nuc_pseudo_cyto_ratio'
            prefix_column_to_extract = 'nuc_pseudo_cyto_int_ratio_ch_'
            prefix_x_label = 'nuc_pseudo_cyto_int_ratio_ch_'
        else:
            pattern = r'^nuc_cyto_int_ratio_ch_\d'
            title_plot  = 'nuc_cyto_ratio'
            prefix_column_to_extract = 'nuc_cyto_int_ratio_ch_'
            prefix_x_label = 'nuc_cyto_int_ratio_ch_'
        
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Plotting
        _, ax = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 5))
        file_name = title_plot +'_'+title_string+'.pdf'
        colors = ['r','g','b','m']
        number_cells = dataframe['cell_id'].nunique()
        for i in range (0,number_color_channels ):
            colum_to_extract = prefix_column_to_extract+str(i)
            int_ratio = np.asarray( [  dataframe.loc[(dataframe['cell_id']==i)][colum_to_extract].values[0]  for i in range(0, number_cells)] )
            if remove_outliers ==True:
                int_ratio =Utilities().remove_outliers( int_ratio,min_percentile=1,max_percentile=99)
            ax[i].hist(x=int_ratio, bins=30, density = True, histtype ='bar',color = colors[i],label = 'spots')
            ax[i].set_xlabel(prefix_x_label+str(i) )
            ax[i].set_ylabel('probability' )
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
        
    def plot_all_distributions (self, dataframe,channels_with_cytosol, channels_with_nucleus,channels_with_FISH,minimum_spots_cluster,output_identification_string ):
        if isinstance(channels_with_FISH, list):
            number_fish_channels = (len(channels_with_FISH))
        else:
            number_fish_channels = 1
        list_file_plots_spot_intensity_distributions =[]
        list_file_plots_distributions =[]
        list_file_plots_cell_size_vs_num_spots =[]
        list_file_plots_cell_intensity_vs_num_spots =[]
        # extracting data for each spot type
        for i in range (number_fish_channels):
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells, nuc_size, cyto_size = Utilities().dataframe_extract_data(dataframe,spot_type=i,minimum_spots_cluster=minimum_spots_cluster)
            file_plots_cell_intensity_vs_num_spots = Plots().plot_cell_intensity_spots(dataframe, number_of_spots_per_cell_nucleus, number_of_spots_per_cell_cytosol,output_identification_string,spot_type=i)
            file_plots_spot_intensity_distributions = Plots().plot_spot_intensity_distributions(dataframe,output_identification_string=output_identification_string,remove_outliers=True,spot_type=i) 
            file_plots_distributions = Plots().plotting_results_as_distributions(number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, ts_size, number_of_TS_per_cell, minimum_spots_cluster, output_identification_string=output_identification_string,spot_type=i)
            file_plots_cell_size_vs_num_spots = Plots().plot_cell_size_spots(channels_with_cytosol, channels_with_nucleus, cell_size, number_of_spots_per_cell, cyto_size, number_of_spots_per_cell_cytosol, nuc_size, number_of_spots_per_cell_nucleus,output_identification_string=output_identification_string,spot_type=i)
            # Appending list of files
            list_file_plots_spot_intensity_distributions.append(file_plots_spot_intensity_distributions)
            list_file_plots_distributions.append(file_plots_distributions)
            list_file_plots_cell_size_vs_num_spots.append(file_plots_cell_size_vs_num_spots)
            list_file_plots_cell_intensity_vs_num_spots.append(file_plots_cell_intensity_vs_num_spots)
            del number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size
            list_files_distributions = [list_file_plots_spot_intensity_distributions,list_file_plots_distributions,list_file_plots_cell_size_vs_num_spots,list_file_plots_cell_intensity_vs_num_spots]
        return list_files_distributions #list_file_plots_spot_intensity_distributions,list_file_plots_distributions,list_file_plots_cell_size_vs_num_spots,list_file_plots_cell_intensity_vs_num_spots
    
    def compare_intensities_spots_interpretation(self, merged_dataframe, list_dataframes, list_number_cells,  list_labels, plot_title_suffix, destination_folder, column_name, remove_extreme_values= True,max_quantile=0.97,color_palete='CMRmap'):
        file_name = 'ch_int_vs_spots_'+plot_title_suffix+'__'+column_name+'.pdf'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        # Detecting the number of columns in the dataset
        my_list = list_dataframes[0].columns
        filtered_list = [elem for elem in my_list if column_name in elem]
        list_column_names = sorted(filtered_list)
        number_color_channels = len(list_column_names)
        # Iterating for each color channel
        y_value_label = 'Spot_Count'
        list_file_names =[]
        for i,column_with_intensity in enumerate(list_column_names):
            x_value_label = 'Channel '+str(i) +' Intensity'
            title_plot='temp__'+str(np.random.randint(1000, size=1)[0])+'_ch_'+str(i)+'_spots.png'
            list_file_names.append(title_plot)
            #column_with_intensity = column_name +str(i)
            list_cell_int = []
            for j in range (len(list_dataframes)):
                list_cell_int.append( Utilities().function_get_df_columns_as_array(df=list_dataframes[j], colum_to_extract=column_with_intensity, extraction_type='values_per_cell')  )
            df_cell_int = Utilities().convert_list_to_df (list_number_cells, list_cell_int, list_labels, remove_extreme_values= remove_extreme_values,max_quantile=max_quantile)
            # This code creates a single column for all conditions and adds a 'location' column.
            df_all_melt = merged_dataframe.melt()
            df_all_melt.rename(columns={'value' : y_value_label}, inplace=True)
            df_int_melt = df_cell_int.melt()
            df_int_melt.rename(columns={'value' : x_value_label}, inplace=True)
            data_frames_list = [df_all_melt, df_int_melt[x_value_label]]
            data_frames = pd.concat(data_frames_list, axis=1)
            data_frames
            # Plotting
            plt.figure(figsize=(5,5))
            sns.set(font_scale = 1.5)
            b= sns.scatterplot( data = data_frames, x = x_value_label, y = y_value_label, hue = 'variable',  alpha = 0.9, palette = color_palete)
            b.set_xlabel(x_value_label)
            b.set_ylabel(y_value_label)
            b.legend(fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.savefig(title_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
            plt.close()
            del b, data_frames, df_int_melt, df_all_melt, df_cell_int
        # Saving a plot with all channels
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (15, 7))
        for i in range(number_color_channels):
            axes[i].imshow(plt.imread(list_file_names[i]))
            os.remove(list_file_names[i])
            axes[i].grid(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(file_name).rename(pathlib.Path().absolute().joinpath(destination_folder,file_name))
        
    def plot_single_cell_all_channels(self, image, df=None, spot_type=0,min_ts_size=4,show_spots=False,image_name=None,microns_per_pixel=None,max_percentile=99.8):
        # Extracting spot localization
        if not (df is None):
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        number_color_channels = image.shape[3]
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 7))
        for i in range(0, number_color_channels):
            temp_image = np.max(image[:,: ,:,i],axis=0)
            max_visualization_value = np.percentile(temp_image,max_percentile)
            min_visualization_value = np.percentile(temp_image, 0)
            axes[i].imshow( temp_image,cmap = 'plasma', vmin=min_visualization_value,vmax=max_visualization_value)
            axes[i].grid(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(r'$_{max}$z (channel '+str(i)+')')
            if (show_spots == True) and (not(df is None)):
            # Plotting spots on image
                for sp in range (number_spots):
                    circle1=plt.Circle((x_spot_locations[sp], y_spot_locations[sp]), 2, color = 'k', fill = False,lw=1)
                    axes[i].add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for ts in range (number_TS):
                        circleTS=plt.Circle((x_TS_locations[ts], y_TS_locations[ts]), 6, color = 'b', fill = False,lw=3)
                        axes[i].add_artist(circleTS)  
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[i].add_artist(scalebar)
        # Saving the image
        if not (image_name is None):               
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    def plot_single_cell(self, image, df, selected_channel, spot_type=0,min_ts_size=4,show_spots=True,image_name=None,microns_per_pixel=None,show_legend = True,max_percentile=99.5,selected_colormap = 'plasma',show_title=True):
        # Extracting spot localization
        y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        # maximum and minimum values to plot
        max_visualization_value = np.percentile(np.max(image[:,: ,:,selected_channel],axis=0),max_percentile)
        min_visualization_value = np.percentile(np.max(image[:,: ,:,selected_channel],axis=0), 0)
        # Section that detects the number of subplots to show
        if show_spots == True:
            number_columns = 2
            x_plot_size =18
        else:
            number_columns = 1
            x_plot_size =9
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_columns, figsize = (x_plot_size, 6))
        if show_spots == True:
            axis_index = axes[0]
        else:
            axis_index = axes
        # Visualizing image only
        axis_index.imshow( np.max(image[:,: ,:,selected_channel],axis=0),cmap = selected_colormap,
                    vmin=min_visualization_value, vmax=max_visualization_value)
        axis_index.grid(False)
        axis_index.set_xticks([])
        axis_index.set_yticks([])
        if show_title == True:
            axis_index.set_title(r'$_{max}$z (channel '+str(selected_channel) +')')
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
            axis_index.add_artist(scalebar)
        # Visualization image with detected spots
        if show_spots == True:
            axes[1].imshow( np.max(image[:,: ,:,selected_channel],axis=0),cmap = selected_colormap,
                            vmin=min_visualization_value, vmax=max_visualization_value)
            axes[1].grid(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_title(r'$_{max}$z channel ('+str(selected_channel) + ') and detected spots')
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[1].add_artist(scalebar)
            if show_spots == True:
                # Plotting spots on image
                for i in range (number_spots):
                    if i < number_spots-1:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1)
                    else:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1, label='Spots = '+str(number_spots))
                    axes[1].add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'b', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'b', fill = False,lw=3, label= 'TS = '+str(number_TS) )
                        axes[1].add_artist(circleTS )
                # showing label with number of spots and ts.
                if show_legend == True: 
                    legend = axes[1].legend(loc='upper right',facecolor= 'white')
                    legend.get_frame().set_alpha(None)
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    def plot_cell_all_z_planes(self, image, image_name=None ):
        number_color_channels = image.shape[3]
        number_z_slices = image.shape[0]
        _, axes = plt.subplots(nrows = number_color_channels , ncols = number_z_slices, figsize = ( number_z_slices*2, 10 ))
        for i in range(0, number_z_slices):
            for j in range(0, number_color_channels):
                temp_image = image[i,: ,:,j]
                max_visualization_value = np.percentile(temp_image,99.5)
                min_visualization_value = np.percentile(temp_image, 0)
                axes[j,i].imshow( temp_image,cmap='plasma', vmin=min_visualization_value,vmax=max_visualization_value)
                axes[j,i].grid(False)
                axes[j,i].set_xticks([])
                axes[j,i].set_yticks([])
                if i ==0:
                    axes[j,i].set_ylabel('Channel '+str(j) )
                if j == 0:
                    axes[j,i].set_title(r'$z_{plane}$ '+str(i) )
            # Saving the image
        if not (image_name is None):             
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    
    def plot_selected_cell_colors(self, image, df, spot_type=0, min_ts_size=None, show_spots=True,use_gaussian_filter = True, image_name=None,microns_per_pixel=None, show_legend=True,list_channel_order_to_plot=[0,1,2], max_percentile=99.8):
        # Extracting spot location
        y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        # Applying Gaussian filter
        if use_gaussian_filter == True:
            filtered_image_with_selected_cell = GaussianFilter(video=image, sigma = 1).apply_filter()
            max_subsection_image_with_selected_cell = np.max(filtered_image_with_selected_cell,axis=0)
        else:
            max_subsection_image_with_selected_cell = np.max(image[:,: ,:,:],axis=0)
        # Converting to int8
        print('max_sub',np.max(max_subsection_image_with_selected_cell))
        subsection_image_with_selected_cell_int8 = Utilities().convert_to_int8(max_subsection_image_with_selected_cell, rescale=True, min_percentile=0.5, max_percentile=max_percentile)
        print('max',np.max(subsection_image_with_selected_cell_int8))
        print('shape',subsection_image_with_selected_cell_int8.shape)
        #print('test', subsection_image_with_selected_cell_int8.shape[2]<3  )
        # padding with zeros the channel dimension.
        while subsection_image_with_selected_cell_int8.shape[2]<3:
            zeros_plane = np.zeros_like(subsection_image_with_selected_cell_int8[:,:,0])
            subsection_image_with_selected_cell_int8 = np.concatenate((subsection_image_with_selected_cell_int8,zeros_plane[:,:,np.newaxis]),axis=2)
        # Plot maximum projection
        if show_spots == True:
            number_columns = 2
            x_plot_size =12
        else:
            number_columns = 1
            x_plot_size =6
        _, axes = plt.subplots(nrows = 1, ncols = number_columns, figsize = (x_plot_size, 6))
        if show_spots == True:
            axis_index = axes[0]
        else:
            axis_index = axes
        # Plotting original image
        axis_index.imshow( subsection_image_with_selected_cell_int8[:,:,list_channel_order_to_plot])
        axis_index.grid(False)
        axis_index.set_xticks([])
        axis_index.set_yticks([])
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
            axis_index.add_artist(scalebar)
        if show_spots == True:
            # Plotting image with detected spots
            axes[1].imshow( subsection_image_with_selected_cell_int8[:,:,list_channel_order_to_plot])
            axes[1].grid(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[1].add_artist(scalebar)
        if show_spots == True:
            # Plotting spots on image
                for i in range (number_spots):
                    if i < number_spots-1:
                        circle=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1)
                    else:
                        circle=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1, label='Spots = '+str(number_spots))
                    axes[1].add_artist(circle)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'y', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'y', fill = False,lw=3, label= 'TS = '+str(number_TS) )
                        axes[1].add_artist(circleTS ) 
                if show_legend == True: 
                    legend = axes[1].legend(loc='upper right',facecolor= 'white')
                    legend.get_frame().set_alpha(None)
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.pdf':
                image_name = image_name+'.pdf'
            plt.savefig(image_name, transparent=False,dpi=1200, bbox_inches = 'tight', format='pdf')
        plt.show()
        return None
    
    
    def plot_complete_fov(self, list_images, df, number_of_selected_image, use_GaussianFilter=True,microns_per_pixel = None,image_name=None,show_cell_ids=True,list_channel_order_to_plot=None,min_percentile=10, max_percentile=99.5):
        df_selected_cell = df.loc[   (df['image_id']==number_of_selected_image)]
        if use_GaussianFilter == True:
            video_filtered = GaussianFilter(video=list_images[number_of_selected_image], sigma = 1).apply_filter()
            max_complete_image = np.max(video_filtered,axis=0)
        else:
            max_complete_image = np.max(list_images[number_of_selected_image],axis=0)
        max_complete_image_int8 = Utilities().convert_to_int8(max_complete_image, rescale=True, min_percentile=min_percentile, max_percentile=max_percentile)    
        # Plot maximum projection
        _, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 15))
        if not (list_channel_order_to_plot is None):
            axes.imshow( max_complete_image_int8[:,:,list_channel_order_to_plot])
        else:
            axes.imshow( max_complete_image_int8[:,:,[2,1,0]])
        axes.grid(False)
        axes.set_xticks([])
        axes.set_yticks([])
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.5,location='lower right',box_color='k',color='w', font_properties=font_props)
            axes.add_artist(scalebar)
        if show_cell_ids == True:
            moving_scale =40 # This parameter moves the label position.
            cell_ids_labels = np.unique(df_selected_cell.loc[ :,'cell_id'].values)
            for _, label in enumerate(cell_ids_labels):
                cell_idx_string = str(label)
                Y_cell_location = df_selected_cell.loc[df_selected_cell['cell_id'] == label, 'nuc_loc_y'].values[0]-moving_scale
                X_cell_location = df_selected_cell.loc[df_selected_cell['cell_id'] == label, 'nuc_loc_x'].values[0]
                if X_cell_location>moving_scale:
                    X_cell_location = X_cell_location-moving_scale   
                axes.text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='w')
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    
    def plot_all_cells_and_spots(self, list_images, complete_dataframe, selected_channel, list_masks_complete_cells= [None], list_masks_nuclei=[None], spot_type=0,list_segmentation_successful=None,min_ts_size=4,image_name=None,microns_per_pixel=None,show_legend = True,show_plot=True,use_max_projection=True):
        # removing images where segmentation was not successful
        if not (list_segmentation_successful is None):
            list_images = [list_images[i] for i in range(len(list_images)) if list_segmentation_successful[i]]
        #Calculating number of subplots 
        number_cells = np.max(complete_dataframe['cell_id'].values)+1
        NUM_COLUMNS = 10
        NUM_ROWS =  math.ceil(number_cells/ NUM_COLUMNS) *2 
        max_size_y_image_size = 800
        y_image_size = np.min((max_size_y_image_size,NUM_ROWS*4))
        # Read the list of masks
        NUM_POINTS_MASK_EDGE_LINE = 100
        if not (list_masks_complete_cells[0] is None):
            list_cell_masks = []
            for _, masks_image in enumerate (list_masks_complete_cells):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_cell_masks.append(tested_mask)
        else:
            list_cell_masks=[None]
        if not (list_masks_nuclei[0] is None):
            list_nuc_masks = []
            for _, masks_image in enumerate (list_masks_nuclei):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_nuc_masks.append(tested_mask)
        else:
            list_nuc_masks=[None]
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (30, y_image_size))
        # Extracting image with cell and specific dataframe
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        # Plotting cells only
        r = 0
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, dataframe=complete_dataframe)
            # maximum and minimum values to plot
            central_z_slice = int(image.shape[0]/2)
            if use_max_projection ==True:
                temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
                #in_focus_image = RemoveOutFocusPlanes.get_frames_in_focus(image)
                #temp_image = np.max(in_focus_image[:,: ,:,selected_channel],axis=0)
                #z_slice =None
            else:
                temp_image = image[central_z_slice,: ,:,selected_channel]
                #z_slice = central_z_slice
            #image_width = temp_image.shape[1]
            #max_int = np.mean(temp_image).astype(int)
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Extracting spot localization
            #y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size,z_slice=z_slice)
            # Plotting
            # Visualizing image only
            axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            #axis_index.set_title('Cell_'+str(cell_id) +' w: '+str(image_width) + ' m: ' + str(max_int))
            axis_index.set_title('Cell_'+str(cell_id) )
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=2
        # Plotting cells with detected spots
        r = 1
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            if not(list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=None, dataframe=complete_dataframe)
            if not(list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=None, dataframe=complete_dataframe)
            # Extracting spot localization
            #y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
            # maximum and minimum values to plot
            central_z_slice = int(image.shape[0]/2)
            if use_max_projection ==True:
                #in_focus_image = RemoveOutFocusPlanes.get_frames_in_focus(image)
                #temp_image = np.max(in_focus_image[:,: ,:,selected_channel],axis=0)
                temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
                z_slice =None
            else:
                temp_image = image[central_z_slice,: ,:,selected_channel]
                z_slice =central_z_slice
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size,z_slice=z_slice)
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Plotting
            axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title('Cell '+str(cell_id) + ' - Detection')
            # plotting the mask if exitsts
            if not( cell_mask is None):
                temp_contour = find_contours(cell_mask, 0.5, fully_connected='high',positive_orientation='high')
                contour = np.asarray(temp_contour[0])
                downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'm', linewidth=1.5) 
            if not (nuc_mask is None):
                temp_contour = find_contours(nuc_mask, 0.5, fully_connected='high',positive_orientation='high')
                contour = np.asarray(temp_contour[0])
                downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'b', linewidth=1.5) 
            # Plotting spots on image
            if number_spots_selected_z >0:
                for i in range (number_spots_selected_z):
                    if i < number_spots_selected_z-1:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.3)
                    else:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.3, label='Spots: '+str(number_spots))
                    axis_index.add_artist(circle1)     
            # Plotting TS
            if number_TS >0:
                for i in range (number_TS):
                    if i < number_TS-1:
                        circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=2.5 )
                    else:
                        circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=2.5, label= 'TS: '+str(number_TS) )
                    axis_index.add_artist(circleTS )
            # showing label with number of spots and ts.
            if (show_legend == True) and (number_spots_selected_z>0): 
                legend = axis_index.legend(loc='upper right',facecolor= 'white',prop={'size': 9})
                legend.get_frame().set_alpha(None)
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=2
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.pdf':
                image_name = image_name+'.pdf' 
            try:
                plt.savefig(image_name, transparent=False,dpi=120, bbox_inches = 'tight', format='pdf')
            except:
                plt.savefig(image_name, transparent=False,dpi=90, bbox_inches = 'tight', format='pdf')
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        return None
    
    
    def plot_all_cells(self, list_images, complete_dataframe, selected_channel, list_masks_complete_cells=[None], list_masks_nuclei=[None],spot_type=0,list_segmentation_successful=None,min_ts_size=4,show_spots=True,image_name=None,microns_per_pixel=None,show_legend = True,show_plot=True):
        # removing images where segmentation was not successful
        if not (list_segmentation_successful is None):
            list_images = [list_images[i] for i in range(len(list_images)) if list_segmentation_successful[i]]
        #Calculating number of subplots 
        number_cells = np.max(complete_dataframe['cell_id'].values)+1
        NUM_COLUMNS = 10
        NUM_ROWS = math.ceil(number_cells/ NUM_COLUMNS)
        max_size_y_image_size = 400
        y_image_size = np.min((max_size_y_image_size,NUM_ROWS*4))
        # Read the list of masks
        NUM_POINTS_MASK_EDGE_LINE = 100
        if not (list_masks_complete_cells[0] is None):
            list_cell_masks = []
            for _, masks_image in enumerate (list_masks_complete_cells):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_cell_masks.append(tested_mask)
        else:
            list_cell_masks=[None]
        if not (list_masks_nuclei[0] is None):
            list_nuc_masks = []
            for _, masks_image in enumerate (list_masks_nuclei):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_nuc_masks.append(tested_mask)
        else:
            list_nuc_masks=[None]
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (30, y_image_size))
        # Extracting image with cell and specific dataframe
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        r = 0
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            if not(list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=None, dataframe=complete_dataframe)
            if not(list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=None, dataframe=complete_dataframe)
            # Extracting spot localization
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z   = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
            # maximum and minimum values to plot
            temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Plotting
            # Visualizing image only
            if show_spots == False:
                axis_index.imshow( temp_image,cmap = 'plasma', vmin=min_visualization_value, vmax=max_visualization_value)
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
                axis_index.set_title('Cell ID '+str(cell_id) )
            # Visualization image with detected spots
            else:
                axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
                axis_index.set_title('Cell ID '+str(cell_id))
                # plotting the mask if exitsts
                if not( cell_mask is None):
                    temp_contour = find_contours(cell_mask, 0.5, fully_connected='high',positive_orientation='high')
                    contour = np.asarray(temp_contour[0])
                    downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                    axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'm', linewidth=1.5) 
                if not (nuc_mask is None):
                    temp_contour = find_contours(nuc_mask, 0.5, fully_connected='high',positive_orientation='high')
                    contour = np.asarray(temp_contour[0])
                    downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                    axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'b', linewidth=1.5) 
                # Plotting spots on image
                if number_spots >0:
                    for i in range (number_spots):
                        if i < number_spots-1:
                            circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.5)
                        else:
                            circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.5, label='Spots: '+str(number_spots))
                        axis_index.add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=3, label= 'TS: '+str(number_TS) )
                        axis_index.add_artist(circleTS )
                # showing label with number of spots and ts.
                if (show_legend == True) and (number_spots>0): 
                    legend = axis_index.legend(loc='upper right',facecolor= 'white',prop={'size': 9})
                    legend.get_frame().set_alpha(None)
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=1
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.pdf':
                image_name = image_name+'.pdf'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        return None
    