import numpy as np                    # For numerical operations with arrays
from scipy.optimize import curve_fit  # For curve fitting


class Intensity():
    '''
    This class is intended to calculate the intensity in the detected spots.

    Parameters

    original_image : NumPy array
        Array of images with dimensions [Z, Y, X, C].
        spot_location_z_y_x 

    '''
    def __init__(self, original_image : np.ndarray, spot_size : int = 5, array_spot_location_z_y_x = [0,0,0] ,  method:str = 'disk_donut'):
        self.original_image = original_image
        self.array_spot_location_z_y_x = array_spot_location_z_y_x
        number_spots = array_spot_location_z_y_x.shape[0]
        self.number_spots = number_spots
        self.method = method
        self.number_channels = original_image.shape[-1]
        self.PIXELS_AROUND_SPOT = 3 # THIS HAS TO BE AN EVEN NUMBER
        if isinstance(spot_size,(int, np.int64,np.int32)) :
            self.spot_size = np.full(number_spots , spot_size)
        elif isinstance(spot_size , (list, np.ndarray) ):
            self.spot_size = spot_size.astype('int')
        else:
            print(type(spot_size))
            raise ValueError("please use integer values for the spot_size or a numpy array with integer values ")
        
    
    def calculate_intensity(self):
        def return_crop(image:np.ndarray, x:int, y:int,spot_range):
            crop_image = image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()
            return crop_image
        
        def gaussian_fit(test_im):
            size_spot = test_im.shape[0]
            image_flat = test_im.ravel()
            def gaussian_function(size_spot, offset, sigma):
                ax = np.linspace(-(size_spot - 1) / 2., (size_spot - 1) / 2., size_spot)
                xx, yy = np.meshgrid(ax, ax)
                kernel =  offset *(np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)))
                return kernel.ravel()
            p0 = (np.min(image_flat) , np.std(image_flat) ) # int(size_spot/2))
            optimized_parameters, _ = curve_fit(gaussian_function, size_spot, image_flat, p0 = p0)
            spot_intensity_gaussian = optimized_parameters[0] # Amplitude
            spot_intensity_gaussian_std = optimized_parameters[1]
            return spot_intensity_gaussian, spot_intensity_gaussian_std
        
        def return_donut(image, spot_size):
            tem_img = image.copy().astype('float')
            center_coordinates = int(tem_img.shape[0]/2)
            range_to_replace = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
            min_index = center_coordinates+range_to_replace[0]
            max_index = (center_coordinates +range_to_replace[-1])+1
            tem_img[min_index: max_index , min_index: max_index] *= np.nan
            removed_center_flat = tem_img.copy().flatten()
            donut_values = removed_center_flat[~np.isnan(removed_center_flat)]
            return donut_values.astype('uint16')
        
        def signal_to_noise_ratio(values_disk,values_donut):
            mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))            
            mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            std_intensity_donut = np.std(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            if std_intensity_donut >0:
                SNR = (mean_intensity_disk-mean_intensity_donut) / std_intensity_donut
            else:
                SNR = 0
            mean_background_int = mean_intensity_donut
            std_background_int = std_intensity_donut
            return SNR, mean_background_int,std_background_int
        
        def disk_donut(values_disk, values_donut,spot_size):
            mean_intensity_disk = np.mean(values_disk.flatten().astype('float'))
            #sum_intensity_disk = np.sum(values_disk.flatten().astype('float'))
            spot_intensity_disk_donut_std = np.std(values_disk.flatten().astype('float'))
            mean_intensity_donut = np.mean(values_donut.flatten().astype('float')) # mean calculation ignoring zeros
            spot_intensity_disk_donut = (mean_intensity_disk*(spot_size**2)) - (mean_intensity_donut*(spot_size**2))
            #spot_intensity_disk_donut = sum_intensity_disk - mean_intensity_donut
            return spot_intensity_disk_donut, spot_intensity_disk_donut_std
        
        # Pre-allocating memory
        intensities_snr = np.zeros((self.number_spots, self.number_channels ))
        intensities_background_mean = np.zeros((self.number_spots, self.number_channels ))
        intensities_background_std = np.zeros((self.number_spots, self.number_channels ))
        intensities_mean = np.zeros((self.number_spots, self.number_channels ))
        intensities_std = np.zeros((self.number_spots, self.number_channels ))
        
        for sp in range(self.number_spots):
            for i in range(0, self.number_channels):
                x_pos = self.array_spot_location_z_y_x[sp,2]
                y_pos = self.array_spot_location_z_y_x[sp,1]
                z_pos = self.array_spot_location_z_y_x[sp,0]
                
                individual_spot_size = self.spot_size[sp]
                spots_range_to_replace = np.linspace(-(individual_spot_size - 1) / 2, (individual_spot_size - 1) / 2, individual_spot_size,dtype=int)
                crop_range_to_replace = np.linspace(-(individual_spot_size+self.PIXELS_AROUND_SPOT - 1) / 2, (individual_spot_size+self.PIXELS_AROUND_SPOT - 1) / 2, individual_spot_size+self.PIXELS_AROUND_SPOT,dtype=int)
                
                crop_with_disk_and_donut = return_crop(self.original_image[z_pos, :, :, i], x_pos, y_pos, spot_range=crop_range_to_replace) # 
                values_disk = return_crop(self.original_image[z_pos, :, :, i], x_pos, y_pos, spot_range=spots_range_to_replace) 
                values_donut = return_donut( crop_with_disk_and_donut,spot_size=individual_spot_size)
                intensities_snr[sp,i], intensities_background_mean[sp,i], intensities_background_std[sp,i]  = signal_to_noise_ratio(values_disk,values_donut) # SNR
                
                if self.method == 'disk_donut':
                    intensities_mean[sp,i], intensities_std[sp,i] = disk_donut(values_disk,values_donut,spot_size=individual_spot_size )
                elif self.method == 'total_intensity':
                    intensities_mean[sp,i] = np.max((0, np.mean(values_disk)))# mean intensity in the crop
                    intensities_std[sp,i] = np.max((0, np.std(values_disk)))# std intensity in the crop
                elif self.method == 'gaussian_fit':
                    intensities_mean[sp,i], intensities_std[sp,i] = gaussian_fit(values_disk)# disk_donut(crop_image, self.disk_size
        return np.round(intensities_mean,4), np.round(intensities_std,4), np.round(intensities_snr,4), np.round(intensities_background_mean,4), np.round(intensities_background_std,4)

