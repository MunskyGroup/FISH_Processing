import numpy as np


class RemoveExtrema():
    '''
    This class is intended to remove extreme values from an image. The format of the image must be [Y, X], [Y, X, C], or [Z, Y, X, C].
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Y, X], [Y, X, C], or [Z, Y, X, C].
    min_percentile : float, optional
        Lower bound to normalize intensity. The default is 1.
    max_percentile : float, optional
        Higher bound to normalize intensity. The default is 99.
    selected_channels : List or None, optional
        Use this option to select a list channels to remove extrema. The default is None and applies the removal of extrema to all the channels.
    '''
    def __init__(self, image:np.ndarray, min_percentile:float = 1, max_percentile:float = 99, selected_channels = None):
        self.image = image
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        if not (type(selected_channels) is list):
            self.selected_channels = [selected_channels]
        else:
            self.selected_channels =selected_channels

    def remove_outliers(self):
        '''
        This method normalizes the values of an image by removing extreme values.
        
        Returns
        
        normalized_image : np.uint16
            Normalized image. Array with dimensions [Y, X, C], [Y, X], or [Z, Y, X, C].
        '''
        normalized_image = np.copy(self.image)
        normalized_image = np.array(normalized_image, 'float32')
        # Normalization code for image with format [Y, X]
        if len(self.image.shape) == 2:
            number_time_points = 1
            number_channels = 1
            normalized_image_temp = normalized_image
            if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                max_val = np.percentile(normalized_image_temp, self.max_percentile)
                min_val = np.percentile(normalized_image_temp, self.min_percentile)
                normalized_image_temp [normalized_image_temp < min_val] = min_val
                normalized_image_temp [normalized_image_temp > max_val] = max_val
                normalized_image_temp [normalized_image_temp < 0] = 0
        # Normalization for image with format [Y, X, C].
        if len(self.image.shape) == 3:
            number_channels   = self.image.shape[2]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    normalized_image_temp = normalized_image[ :, :, index_channels]
                    if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                        max_val = np.percentile(normalized_image_temp, self.max_percentile)
                        min_val = np.percentile(normalized_image_temp, self.min_percentile)
                        normalized_image_temp [normalized_image_temp < min_val] =  min_val
                        normalized_image_temp [normalized_image_temp > max_val] = max_val
                        normalized_image_temp [normalized_image_temp < 0] = 0
        # Normalization for image with format [Z, Y, X, C].
        if len(self.image.shape) == 4:
            number_timepoints, number_channels   = self.image.shape[0], self.image.shape[3]
            for index_channels in range (number_channels):
                if (index_channels in self.selected_channels) or (self.selected_channels is None) :
                    for index_time in range (number_time_points):
                        normalized_image_temp = normalized_image[index_time, :, :, index_channels]
                        if not np.max(normalized_image_temp) == 0: # this section detect that the channel is not empty to perform the normalization.
                            max_val = np.percentile(normalized_image_temp, self.max_percentile)
                            min_val = np.percentile(normalized_image_temp, self.min_percentile)
                            normalized_image_temp [normalized_image_temp < min_val] = min_val
                            normalized_image_temp [normalized_image_temp > max_val] = max_val
                            normalized_image_temp [normalized_image_temp < 0] = 0
        return np.array(normalized_image, 'uint16')
