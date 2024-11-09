# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detect spots in 2-d and 3-d.
"""

import warnings

import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

import bigfish.stack as stack

from .utils_BF import get_object_radius_pixel
from .utils_BF import get_breaking_point

from .utils_BF import convert_spot_coordinates
from .utils_BF import get_object_radius_pixel
from .utils_BF import get_object_radius_nm
from .utils_BF import build_reference_spot
from .utils_BF import get_spot_volume
from .utils_BF import get_spot_surface
from .utils_BF import compute_snr_spots
from .utils_BF import get_breaking_point

from skimage.measure import regionprops
from skimage.measure import label


# ### Main function ###

def detect_spots(
        images,
        threshold=None,
        remove_duplicate=True,
        return_threshold=False,
        voxel_size=None,
        spot_radius=None,
        log_kernel_size=None,
        minimum_distance=None):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    #. We smooth the image with a LoG filter.
    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.
    #. We remove local peaks under a threshold.
    #. We keep only one pixel coordinate per detected spot.

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Image (or list of images) with shape (z, y, x) or (y, x). If several
        images are provided, the same threshold is applied.
    threshold : int, float or None
        A threshold to discriminate relevant spots from noisy blobs. If None,
        optimal threshold is selected automatically. If several images are
        provided, one optimal threshold is selected for all the images.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    return_threshold : bool
        Return the threshold used to detect spots.
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.
    log_kernel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of the LoG kernel. It equals the standard deviation (in pixels)
        used for the gaussian kernel (one for each dimension). One value per
        spatial dimension (zyx or yx dimensions). If it's a scalar, the same
        standard deviation is applied to every dimensions. If None, we estimate
        it with the voxel size and spot radius.
    minimum_distance : int, float, Tuple(int, float), List(int, float) or None
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions. If None, we estimate it with the voxel size and spot
        radius.

    Returns
    -------
    spots : List[np.ndarray] or np.ndarray, np.int64
        Coordinates (or list of coordinates) of the spots with shape
        (nb_spots, 3) or (nb_spots, 2), for 3-d or 2-d images respectively.
    threshold : int or float
        Threshold used to discriminate spots from noisy blobs.

    """
    # check parameters
    stack.check_parameter(
        threshold=(int, float, type(None)),
        remove_duplicate=bool,
        return_threshold=bool,
        voxel_size=(int, float, tuple, list, type(None)),
        spot_radius=(int, float, tuple, list, type(None)),
        log_kernel_size=(int, float, tuple, list, type(None)),
        minimum_distance=(int, float, tuple, list, type(None)))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        stack.check_array(
            images,
            ndim=[2, 3],
            dtype=[np.uint8, np.uint16, np.float32, np.float64])
        ndim = images.ndim
        images = [images]
        is_list = False
    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(
                image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
        is_list = True

    # check consistency between parameters - detection with voxel size and
    # spot radius
    if (voxel_size is not None and spot_radius is not None
            and log_kernel_size is None and minimum_distance is None):
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError("'voxel_size' must be a scalar or a sequence "
                                 "with {0} elements.".format(ndim))
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError("'spot_radius' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            spot_radius = (spot_radius,) * ndim
        log_kernel_size = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        minimum_distance = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)

    # check consistency between parameters - detection with kernel size and
    # minimal distance
    elif (voxel_size is None and spot_radius is None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # check consistency between parameters - detection in priority with kernel
    # size and minimal distance
    elif (voxel_size is not None and spot_radius is not None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # missing parameters
    else:
        raise ValueError("One of the two pairs of parameters ('voxel_size', "
                         "'spot_radius') or ('log_kernel_size', "
                         "'minimum_distance') should be provided.")

    # detect spots
    if return_threshold:
        spots, threshold = _detect_spots_from_images(
            images,
            threshold=threshold,
            remove_duplicate=remove_duplicate,
            return_threshold=return_threshold,
            log_kernel_size=log_kernel_size,
            min_distance=minimum_distance)
    else:
        spots = _detect_spots_from_images(
            images,
            threshold=threshold,
            remove_duplicate=remove_duplicate,
            return_threshold=return_threshold,
            log_kernel_size=log_kernel_size,
            min_distance=minimum_distance)

    # format results
    if not is_list:
        spots = spots[0]

    # return threshold or not
    if return_threshold:
        return spots, threshold
    else:
        return spots


def _detect_spots_from_images(
        images,
        threshold=None,
        remove_duplicate=True,
        return_threshold=False,
        log_kernel_size=None,
        min_distance=None):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    #. We smooth the image with a LoG filter.
    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.
    #. We remove local peaks under a threshold.
    #. We keep only one pixel coordinate per detected spot.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images with shape (z, y, x) or (y, x). The same threshold is
        applied to every images.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs. If None,
        optimal threshold is selected automatically. If several images are
        provided, one optimal threshold is selected for all the images.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    return_threshold : bool
        Return the threshold used to detect spots.
    log_kernel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of the LoG kernel. It equals the standard deviation (in pixels)
        used for the gaussian kernel (one for each dimension). One value per
        spatial dimension (zyx or yx dimensions). If it's a scalar, the same
        standard deviation is applied to every dimensions. If None, we estimate
        it with the voxel size and spot radius.
    min_distance : int, float, Tuple(int, float), List(int, float) or None
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions. If None, we estimate it with the voxel size and spot
        radius.

    Returns
    -------
    all_spots : List[np.ndarray], np.int64
        List of spot coordinates with shape (nb_spots, 3) or (nb_spots, 2),
        for 3-d or 2-d images respectively.
    threshold : int or float
        Threshold used to discriminate spots from noisy blobs.

    """
    # initialization
    n = len(images)

    # apply LoG filter and find local maximum
    images_filtered = []
    pixel_values = []
    masks = []
    for image in images:
        # filter image
        image_filtered = stack.log_filter(image, log_kernel_size)
        images_filtered.append(image_filtered)

        # get pixels value
        pixel_values += list(image_filtered.ravel())

        # find local maximum
        mask_local_max = local_maximum_detection(image_filtered, min_distance)
        masks.append(mask_local_max)

    # get optimal threshold if necessary based on all the images
    if threshold is None:

        # get threshold values we want to test
        thresholds = _get_candidate_thresholds(pixel_values)

        # get spots count and its logarithm
        all_value_spots = []
        minimum_threshold = float(thresholds[0])
        for i in range(n):
            image_filtered = images_filtered[i]
            mask_local_max = masks[i]
            spots, mask_spots = spots_thresholding(
                image_filtered, mask_local_max,
                threshold=minimum_threshold,
                remove_duplicate=False)
            value_spots = image_filtered[mask_spots]
            all_value_spots.append(value_spots)
        all_value_spots = np.concatenate(all_value_spots)
        thresholds, count_spots = _get_spot_counts(thresholds, all_value_spots)

        # select threshold where the kink of the distribution is located
        if count_spots.size > 0:
            threshold, _, _ = get_breaking_point(thresholds, count_spots)

    # detect spots
    all_spots = []
    for i in range(n):

        # get images and masks
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]

        # detection
        spots, _ = spots_thresholding(
            image_filtered, mask_local_max, threshold, remove_duplicate)
        all_spots.append(spots)

    # return threshold or not
    if return_threshold:
        return all_spots, threshold
    else:
        return all_spots


# ### LoG spot detection ###

def local_maximum_detection(image, min_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.

    Several connected pixels can have the same value. In such a case, the
    local maximum is not unique.

    In order to make the detection robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example).

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    min_distance : int, float, Tuple(int, float), List(int, float)
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(min_distance=(int, float, tuple, list))

    # compute the kernel size (centered around our pixel because it is uneven)
    if isinstance(min_distance, (tuple, list)):
        if len(min_distance) != image.ndim:
            raise ValueError(
                "'min_distance' should be a scalar or a sequence with one "
                "value per dimension. Here the image has {0} dimensions and "
                "'min_distance' {1} elements.".format(image.ndim,
                                                      len(min_distance)))
    else:
        min_distance = (min_distance,) * image.ndim
    min_distance = np.ceil(min_distance).astype(image.dtype)
    kernel_size = 2 * min_distance + 1

    # apply maximum filter to the original image
    image_filtered = ndi.maximum_filter(image, size=kernel_size)

    # we keep the pixels with the same value before and after the filtering
    mask = image == image_filtered

    return mask


def spots_thresholding(
        image,
        mask_local_max,
        threshold,
        remove_duplicate=True):
    """Filter detected spots and get coordinates of the remaining spots.

    In order to make the thresholding robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example). If
    the local maximum is not unique (it can happen if connected pixels have
    the same value), a connected component algorithm is applied to keep only
    one coordinate per spot.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float, int or None
        A threshold to discriminate relevant spots from noisy blobs. If None,
        detection is aborted with a warning.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(
        mask_local_max,
        ndim=[2, 3],
        dtype=bool)
    stack.check_parameter(
        threshold=(float, int, type(None)),
        remove_duplicate=bool)

    if threshold is None:
        mask = np.zeros_like(image, dtype=bool)
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        warnings.warn("No spots were detected (threshold is {0})."
                      .format(threshold),
                      UserWarning)
        return spots, mask

    # remove peak with a low intensity
    mask = (mask_local_max & (image > threshold))
    if mask.sum() == 0:
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        return spots, mask

    # make sure we detect only one coordinate per spot
    if remove_duplicate:
        # when several pixels are assigned to the same spot, keep the centroid
        cc = label(mask)
        local_max_regions = regionprops(cc)
        spots = []
        for local_max_region in local_max_regions:
            spot = np.array(local_max_region.centroid)
            spots.append(spot)
        spots = np.stack(spots).astype(np.int64)

        # built mask again
        mask = np.zeros_like(mask)
        mask[spots[:, 0], spots[:, 1]] = True

    else:
        # get peak coordinates
        spots = np.nonzero(mask)
        spots = np.column_stack(spots)

    # case where no spots were detected
    if spots.size == 0:
        warnings.warn("No spots were detected (threshold is {0})."
                      .format(threshold),
                      UserWarning)

    return spots, mask


# ### Threshold selection ###

def automated_threshold_setting(image, mask_local_max):
    """Automatically set the optimal threshold to detect spots.

    In order to make the thresholding robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example). The
    optimal threshold is selected based on the spots distribution. The latter
    should have an elbow curve discriminating a fast decreasing stage from a
    more stable one (a plateau).

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    Returns
    -------
    optimal_threshold : int
        Optimal threshold to discriminate spots from noisy blobs.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(
        mask_local_max,
        ndim=[2, 3],
        dtype=bool)

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(image.ravel())

    # get spots count and its logarithm
    first_threshold = float(thresholds[0])
    spots, mask_spots = spots_thresholding(
        image, mask_local_max, first_threshold, remove_duplicate=False)
    value_spots = image[mask_spots]
    thresholds, count_spots = _get_spot_counts(thresholds, value_spots)

    # select threshold where the break of the distribution is located
    if count_spots.size > 0:
        optimal_threshold, _, _ = get_breaking_point(thresholds, count_spots)

    # case where no spots were detected
    else:
        optimal_threshold = None

    return optimal_threshold


def _get_candidate_thresholds(pixel_values):
    """Choose the candidate thresholds to test for the spot detection.

    Parameters
    ----------
    pixel_values : np.ndarray
        Pixel intensity values of the image.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.

    """
    # choose appropriate thresholds candidate
    start_range = 0
    end_range = int(np.percentile(pixel_values, 99.9999))
    if end_range < 100:
        thresholds = np.linspace(start_range, end_range, num=100)
    else:
        thresholds = [i for i in range(start_range, end_range + 1)]
    thresholds = np.array(thresholds)

    return thresholds


def _get_spot_counts(thresholds, value_spots):
    """Compute and format the spots count function for different thresholds.

    Parameters
    ----------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    value_spots : np.ndarray
        Pixel intensity values of all spots.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    count_spots : np.ndarray, np.float64
        Spots count function (log scale).

    """
    # count spots for each threshold
    count_spots = np.log([np.count_nonzero(value_spots > t)
                          for t in thresholds])
    count_spots = stack.centered_moving_average(count_spots, n=5)

    # the tail of the curve unnecessarily flatten the slop
    count_spots = count_spots[count_spots > 2]
    thresholds = thresholds[:count_spots.size]

    return thresholds, count_spots

def compute_snr_spots(image, spots, voxel_size, spot_radius, display_plots: bool = False):
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(
        spots,
        ndim=2,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.uint16)
    radius_background = np.ceil(radius_background_).astype(np.uint16)

    snr_spots = []

    # Loop over each spot
    for spot in spots:
        # Extract spot coordinates
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx

        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            # Compute max signal for the current spot
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            # Extract background region
            spot_background_, _ = get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[:, edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            # For 2D images
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # Compute mean and standard deviation of the background
        mean_background = np.mean(spot_background)
        std_background = np.std(spot_background)

        # Compute SNR
        snr = (max_signal - mean_background) / (std_background + 1e-8)  # Add small value to avoid division by zero
        snr_spots.append(snr)

    # Returns SNRs for further analysis
    return snr_spots


def get_elbow_values(
        images,
        voxel_size=None,
        spot_radius=None,
        log_kernel_size=None,
        minimum_distance=None,
        use_snr=False,
        display_pre_snr_num_spots=False,
        display_snr_v_threshold=False,
        thresholds_to_test=None):
    """Get values to plot the elbow curve used to automatically set the
    threshold to detect spots.

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Image (or list of images) with shape (z, y, x) or (y, x). If several
        images are provided, the same threshold is applied.
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.
    log_kernel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of the LoG kernel. It equals the standard deviation (in pixels)
        used for the gaussian kernel (one for each dimension). One value per
        spatial dimension (zyx or yx dimensions). If it's a scalar, the same
        standard deviation is applied to every dimensions. If None, we estimate
        it with the voxel size and spot radius.
    minimum_distance : int, float, Tuple(int, float), List(int, float) or None
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions. If None, we estimate it with the voxel size and spot
        radius.
    use_snr : bool
        Compute the signal-to-noise ratio (SNR) for each spot and each
        threshold. The SNR is computed based on the spot coordinates.    
    display_pre_snr_num_spots : bool
        Display the number of spots and the SNR computed for each spot for
        different thresholds.
    display_snr_v_threshold : bool
        Display the SNR computed for each spot for different thresholds.
    thresholds_to_test : np.ndarray or None
        Threshold values to test. If None, we use a range of thresholds.    

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    count_spots : np.ndarray, np.float64
        Spots count (log scale).
    threshold : float or None
        Threshold automatically set.
    snr_spot_values : np.ndarray, np.float64
        Signal-to-noise ratio (SNR) computed for every spots.    

    """
    # check parameters
    stack.check_parameter(
        voxel_size=(int, float, tuple, list, type(None)),
        spot_radius=(int, float, tuple, list, type(None)),
        log_kernel_size=(int, float, tuple, list, type(None)),
        minimum_distance=(int, float, tuple, list, type(None)))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        stack.check_array(
            images,
            ndim=[2, 3],
            dtype=[np.uint8, np.uint16, np.float32, np.float64])
        ndim = images.ndim
        images = [images]
        n = 1
    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(
                image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
        n = len(images)

    # check consistency between parameters - detection with voxel size and
    # spot radius
    if (voxel_size is not None and spot_radius is not None
            and log_kernel_size is None and minimum_distance is None):
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError(
                    "'voxel_size' must be a scalar or a sequence "
                    "with {0} elements.".format(ndim))
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError("'spot_radius' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            spot_radius = (spot_radius,) * ndim

        log_kernel_size = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        minimum_distance = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)

    # check consistency between parameters - detection with kernel size and
    # minimal distance
    elif (voxel_size is None and spot_radius is None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError(
                    "'minimum_distance' must be a scalar or a "
                    "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # check consistency between parameters - detection in priority with kernel
    # size and minimal distance
    elif (voxel_size is not None and spot_radius is not None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError(
                    "'minimum_distance' must be a scalar or a "
                    "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # missing parameters
    else:
        raise ValueError(
            "One of the two pairs of parameters ('voxel_size', "
            "'spot_radius') or ('log_kernel_size', "
            "'minimum_distance') should be provided.")

    # apply LoG filter and find local maximum
    images_filtered = []
    pixel_values = []
    masks = []
    for image in images:
        # filter image
        image_filtered = stack.log_filter(image, log_kernel_size)
        images_filtered.append(image_filtered)

        # get pixels value
        pixel_values += list(image_filtered.ravel())

        # find local maximum
        mask_local_max = local_maximum_detection(
            image_filtered, minimum_distance)
        masks.append(mask_local_max)

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(pixel_values)

    # get spots count and its logarithm
    all_value_spots = []
    minimum_threshold = float(thresholds[0])
    for i in range(n):
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]
        spots, mask_spots = spots_thresholding(
            image_filtered, mask_local_max,
            threshold=minimum_threshold,
            remove_duplicate=False)
        value_spots = image_filtered[mask_spots]
        all_value_spots.append(value_spots)
    all_value_spots = np.concatenate(all_value_spots)
    thresholds, count_spots = _get_spot_counts(
        thresholds, all_value_spots)

    # select threshold where the kink of the distribution is located
    if count_spots.size > 0:
        optimal_threshold, _, _ = get_breaking_point(thresholds, count_spots)
    else:
        optimal_threshold = None
    # print optimal threshold and number of spots detected before snr calculation 
    if display_pre_snr_num_spots:   
        print(f"Optimal threshold: {optimal_threshold}")
        print(f"Number of spots detected before snr calculation: {len(all_value_spots)}")    

    if use_snr:
        # calculate snr values
        snr_spot_values = []
        thresholds_to_test = thresholds_to_test = [] if thresholds_to_test is not None else None
        if optimal_threshold is not None:
            thresholds_to_test = [
                optimal_threshold * 0.9,
                optimal_threshold * 0.8,
                optimal_threshold * 1.1,
                optimal_threshold * 1.2,
                optimal_threshold * 1.4
            ]
        else:
            thresholds_to_test = []
        for threshold in thresholds_to_test:
            all_snr = []
            for i in range(n):
                image_filtered = images_filtered[i]
                mask_local_max = masks[i]
                spots, mask_spots = spots_thresholding(
                    image_filtered, mask_local_max, threshold, remove_duplicate=False)
                snr_spots = compute_snr_spots(
                    image_filtered, spots, voxel_size, spot_radius)
                all_snr += snr_spots
            snr_spot_values.append(np.median(all_snr))

        # Plot scatter plots of snr vs threshold for each threshold tested
        if display_snr_v_threshold:
            plt.scatter(thresholds_to_test, snr_spot_values)
            plt.xlabel('Threshold')
            plt.ylabel('SNR')
            plt.title('SNR vs. Threshold')
            plt.show()
            
    return thresholds, count_spots, optimal_threshold, snr_spot_values, thresholds_to_test
# Visualization code
def plot_elbow_snr(thresholds, count_spots, snr_values, optimal_threshold=None):
    """Plot elbow curves for spot count and SNR to visualize optimal threshold."""
    fig, ax1 = plt.subplots()

    # Plot spot count on left y-axis
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Spot Count', color='tab:blue')
    ax1.plot(thresholds, count_spots, label='Spot Count', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot SNR on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('SNR', color='tab:orange')
    ax2.plot(thresholds, snr_values, label='SNR', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Highlight the optimal threshold
    if optimal_threshold:
        ax1.axvline(x=optimal_threshold, color='green', linestyle='--', label='Optimal Threshold')

    # Add a legend and show plot
    fig.tight_layout()
    plt.title("Spot Count and SNR vs. Threshold")
    plt.show()