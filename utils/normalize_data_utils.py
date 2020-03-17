import numpy as np


"""
NOTE : THESE FUNCTIONS ARE IN PROGRESS, THEY WORK BUT THE MATH NEEDS TO BE QUADRUPLE TESTED
"""


def get_brain_region(brain_slice):
    """
    Returns a list of all non-zero values in slice
    """
    brain_region = []
    for i in range(brain_slice.shape[0]):
        for j in range(brain_slice.shape[1]):
            if brain_slice[i,j]:
                brain_region.append(brain_slice[i,j])
    return brain_region


def mean_standard_norm_slice(brain_slice):
    """
    Normalize a single slice
    """
    normalized_slice = brain_slice.copy()
    brain_region_intensities = get_brain_region(brain_slice)
    # check if slice has any brain region
    if len(brain_region_intensities):
        intensity_mean         = np.mean(brain_region_intensities)
        intensity_standard_dev = np.std(brain_region_intensities)
        # don't perform any calculations if std dev == 0. 
        # In this case all brain region values are the same
        if not intensity_standard_dev:
            return normalized_slice
        # apply normalization algo to non-zero values and fill normalized_brain_volume array
        # TO DO: what to do when standard deviation is zero
        for i in range(brain_slice.shape[0]):
            for j in range(brain_slice.shape[1]):
                if brain_slice[i,j]:
                    normalized_slice[i,j] = (brain_slice[i,j] - intensity_mean) / intensity_standard_dev        
    
    return normalized_slice


def mean_standard_norm_volume(brain_volume_data):
    '''
    This function returns a normalized brain volume : same format as an unprocessed volume
    '''
    normalized_data = []
    # iterate through each group of slices, normalize each sice and add it to the dictionary
    for slice_group in brain_volume_data:
        v_slice = {"flair": mean_standard_norm_slice(slice_group["flair"]), 
                    "t1"  : mean_standard_norm_slice(slice_group["t1"]), 
                    "t1ce": mean_standard_norm_slice(slice_group["t1ce"]), 
                    "t2"  : mean_standard_norm_slice(slice_group["t2"]), 
                    "seg" : mean_standard_norm_slice(slice_group["seg"]),
                    "s_id": slice_group["s_id"]}
        normalized_data.append(v_slice)
    return normalized_data