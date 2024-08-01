import multiprocessing  # For CPU count

import numpy as np  # For numerical operations with arrays
from joblib import Parallel, delayed  # For parallel processing
from skimage import img_as_uint  # For converting images to uint16 format
from skimage.filters import gaussian  # For applying Gaussian filter


class GaussianFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [Z, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters

    video : NumPy array
        Array of images with dimensions [Z, Y, X, C].
    sigma : float, optional
        Sigma value for the gaussian filter. The default is 1.
    

    '''
    def __init__(self, video:np.ndarray, sigma:float = 1):
        # Making the values for the filters are odd numbers
        self.video = video
        self.sigma = sigma
        self.NUMBER_OF_CORES = multiprocessing.cpu_count()
    def apply_filter(self):
        '''
        This method applies high and low bandpass filters to the video.

        Returns
        
        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        video_bp_filtered_float = np.zeros_like(self.video, dtype = np.float64)
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                if np.max(self.video[index_time, :, :, index_channels])>0:
                    video_bp_filtered_float[index_time, :, :, index_channels] = gaussian(self.video[index_time, :, :, index_channels], self.sigma)
        # temporal function that converts floats to uint
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
            video_filtered[:,:,:,index_channels] = np.asarray(init_video)
        return video_filtered # video_bp_filtered_float # img_as_uint(video_bp_filtered_float)


