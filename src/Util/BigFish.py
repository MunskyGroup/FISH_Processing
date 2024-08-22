import bigfish.detection as detection
import bigfish.plot as plot
import bigfish.stack as stack
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class BigFISH:
    '''
    This class is intended to detect spots in FISH images using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. The format of the image must be  [Z, Y, X, C].
    
    Parameters
    
    The description of the parameters is taken from `Big-FISH <https://github.com/fish-quant/big-fish>`_ BSD 3-Clause License. Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C]  or [Y, X, C].
    FISH_channel : int
        Specific channel with FISH spots that are used for the quantification
    voxel_size_z : int, optional
        Height of a voxel, along the z axis, in nanometers. The default is 300.
    voxel_size_yx : int, optional
        Size of a voxel on the yx plan in nanometers. The default is 150.
    psf_z : int, optional
        Theoretical size of the PSF emitted by a spot in the z plan, in nanometers. The default is 350.
    psf_yx : int, optional
        Theoretical size of the PSF emitted by a spot in the yx plan in nanometers.
    cluster_radius : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    show_plots : bool, optional
        If True shows a 2D maximum projection of the image and the detected spots. The default is False
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

    def __init__(self, image, FISH_channel, voxel_size_z=300, voxel_size_yx=103, psf_z=350, psf_yx=150,
                 cluster_radius=350, minimum_spots_cluster=4, show_plots=False, image_name=None, save_all_images=False,
                 display_spots_on_multiple_z_planes=False, use_log_filter_for_spot_detection=True,
                 threshold_for_spot_detection=None, save_files=True, bigfish_alpha: float = 0.5,
                 bigfish_beta: float = 1, bigfish_gamma: float = 5,
                 **kwargs):
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=0)
        self.image = image
        self.FISH_channel = FISH_channel
        self.voxel_size_z = voxel_size_z
        self.voxel_size_yx = voxel_size_yx
        self.psf_z = psf_z
        self.psf_yx = psf_yx
        self.cluster_radius = cluster_radius
        self.minimum_spots_cluster = minimum_spots_cluster
        self.show_plots = show_plots
        self.image_name = image_name
        self.save_all_images = save_all_images  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection = use_log_filter_for_spot_detection
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.decompose_dense_regions = True
        self.save_files = save_files
        self.bigfish_alpha = bigfish_alpha
        self.bigfish_beta = bigfish_beta
        self.bigfish_gamma = bigfish_gamma

    def detect(self):
        '''
        This method is intended to detect RNA spots in the cell and Transcription Sites (Clusters) using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
        
        Returns
        
        clusterDetectionCSV : np.int64 Array with shape (nb_clusters, 5) or (nb_clusters, 4). 
            One coordinate per dimension for the centroid of the cluster (zyx or yx coordinates), the number of spots detected in the clusters, and its index.
        spotDetectionCSV :  np.int64 with shape (nb_spots, 4) or (nb_spots, 3).
            Coordinates of the detected spots. One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, the value is -1.
        '''
        # Setting the colormap
        mpl.rc('image', cmap='viridis')
        rna = self.image[:, :, :, self.FISH_channel]
        # Calculating Sigma with  the parameters for the PSF.
        spot_radius_px = detection.get_object_radius_pixel(
            voxel_size_nm=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx),
            object_radius_nm=(self.psf_z, self.psf_yx, self.psf_yx), ndim=3)
        sigma = spot_radius_px
        # print('sigma_value (z,y,x) =', sigma)
        ## SPOT DETECTION
        if self.use_log_filter_for_spot_detection:
            try:
                rna_filtered = stack.log_filter(rna, sigma)  # LoG filter
            except ValueError:
                print('Error during the log filter calculation, try using larger parameters values for the psf')
                rna_filtered = stack.remove_background_gaussian(rna, sigma)
        else:
            rna_filtered = stack.remove_background_gaussian(rna, sigma)
        # Automatic threshold detection.
        mask = detection.local_maximum_detection(rna_filtered, min_distance=sigma)  # local maximum detection
        if not (self.threshold_for_spot_detection is None):
            threshold = self.threshold_for_spot_detection
        else:
            threshold = detection.automated_threshold_setting(rna_filtered, mask)  # thresholding
        # print('Int threshold used for the detection of spots: ',threshold )
        spots, _ = detection.spots_thresholding(rna_filtered, mask, threshold, remove_duplicate=True)
        # Decomposing dense regions
        if self.decompose_dense_regions:
            try:
                # print(self.bigfish_alpha, '++++++++++++++++++++++++++++++++++++++++++++++++++')
                spots_post_decomposition, _, _ = detection.decompose_dense(image=rna,
                                                                           spots=spots,
                                                                           voxel_size=(
                                                                               self.voxel_size_z, self.voxel_size_yx,
                                                                               self.voxel_size_yx),
                                                                           spot_radius=(
                                                                               self.psf_z, self.psf_yx, self.psf_yx),
                                                                           alpha=self.bigfish_alpha,  # alpha impacts the number of spots per candidate region
                                                                           beta=self.bigfish_beta,  # beta impacts the number of candidate regions to decompose
                                                                           gamma=self.bigfish_gamma)  # gamma the filtering step to denoise the image
            except:
                spots_post_decomposition = spots
        else:
            spots_post_decomposition = spots
            # print('Error during step: detection.decompose_dense ')
        ### CLUSTER DETECTION
        spots_post_clustering, clusters = detection.detect_clusters(spots_post_decomposition,
                                                                    voxel_size=(self.voxel_size_z, self.voxel_size_yx,
                                                                                self.voxel_size_yx),
                                                                    radius=self.cluster_radius,
                                                                    nb_min_spots=self.minimum_spots_cluster)

        # remove spots that are part of a cluster
        spots_post_clustering = spots_post_clustering[spots_post_clustering[:, -1] == -1]

        # Saving results with new variable names
        spotDetectionCSV = spots_post_clustering
        clusterDetectionCSV = clusters
        ## PLOTTING
        try:
            if self.save_files:
                if self.image_name is not None:
                    path_output_elbow = str(self.image_name) + '__elbow_' + '_ch_' + str(self.FISH_channel) + '.png'
                    plot.plot_elbow(rna,
                                    voxel_size=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx),
                                    spot_radius=(self.psf_z, self.psf_yx, self.psf_yx),
                                    path_output=path_output_elbow, show=bool(self.show_plots))
                    if self.show_plots:
                        plt.show()
                    else:
                        plt.close()
        except:
            print('not showing elbow plot')
        central_slice = rna.shape[0] // 2
        if self.save_all_images:
            range_plot_images = range(0, rna.shape[0])
        else:
            range_plot_images = range(central_slice, central_slice + 1)
        for i in range_plot_images:
            if i == central_slice and self.show_plots:
                print('Z-Slice: ', str(i))
            image_2D = rna_filtered[i, :, :]
            if 1 < i < rna.shape[0] - 1:
                if self.display_spots_on_multiple_z_planes:
                    # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane-1
                    clusters_to_plot = clusters[(clusters[:, 0] >= i - 1) & (clusters[:, 0] <= i + 2)]
                    spots_to_plot = spots_post_decomposition[
                        (spots_post_decomposition[:, 0] >= i - 1) & (spots_post_decomposition[:, 0] <= i + 1)]
                else:
                    clusters_to_plot = clusters[clusters[:, 0] == i]
                    spots_to_plot = spots_post_decomposition[spots_post_decomposition[:, 0] == i]
            else:
                clusters_to_plot = clusters[clusters[:, 0] == i]
                spots_to_plot = spots_post_decomposition[spots_post_decomposition[:, 0] == i]
            if self.save_all_images:
                path_output = str(self.image_name) + '_ch_' + str(self.FISH_channel) + '_slice_' + str(i) + '.png'
            else:
                path_output = str(self.image_name) + '_ch_' + str(self.FISH_channel) + '.png'

            if (self.image_name is not None) and (i == central_slice) and (
                    self.show_plots):  # saving only the central slice
                show_figure_in_cli = True
            else:
                show_figure_in_cli = False
            if not (self.image_name is None):
                if self.save_files:
                    try:
                        plot.plot_detection(image_2D,
                                            spots=[spots_to_plot, clusters_to_plot[:, :3]],
                                            shape=["circle", "polygon"],
                                            radius=[3, 6],
                                            color=["orangered", "blue"],
                                            linewidth=[1, 1],
                                            fill=[False, False],
                                            framesize=(12, 7),
                                            contrast=True,
                                            rescale=True,
                                            show=show_figure_in_cli,
                                            path_output=path_output)
                    except:
                        pass
                    if self.show_plots:
                        plt.show()
                    else:
                        plt.close()
            del spots_to_plot, clusters_to_plot

        return [spotDetectionCSV, clusterDetectionCSV], rna_filtered, threshold
