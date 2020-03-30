from utils import hgg_utils
from utils import normalize_data_utils as norm
import numpy as np
import nibabel as nib
import pathlib
from tqdm import tqdm

"""
NOTE : THESE FUNCTIONS ARE IN PROGRESS, THEY WORK BUT THE MATH NEEDS TO BE QUADRUPLE CHECKED
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


def scale_slice(brain_slice, scale_factor=None):
    '''
    Scales each pixel by given value
    If no scale value is given, uses (1.0/max_pixel_value)
    '''
    normalized_slice = brain_slice.copy()
    if scale_factor is None:
        max_value = brain_slice.max()
        if not max_value:
            # return original slice if all zero value
            return normalized_slice
        else:
            scale_factor = (1.0/max_value)
    return (normalized_slice * scale_factor)


def scale_volume(brain_volume_data, scale_factor=None):
    '''
    This function scales every pixel value by desired factor
    '''
    normalized_data = []
    # iterate through each group of slices, scale pixel values
    for slice_group in brain_volume_data:
        v_slice = {"flair": scale_slice(slice_group["flair"], scale_factor), 
                    "t1"  : scale_slice(slice_group["t1"], scale_factor), 
                    "t1ce": scale_slice(slice_group["t1ce"], scale_factor), 
                    "t2"  : scale_slice(slice_group["t2"], scale_factor), 
                    "seg" : scale_slice(slice_group["seg"], scale_factor),
                    "s_id": slice_group["s_id"]}
        normalized_data.append(v_slice)
    return normalized_data


def download_scaled_dataset(scale_value=None):
    '''
    If scale value is None uses (1.0/slices_max_pixel_value)
    Creates a standardized dataset in dir : .../MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/scaled_hgg
    '''
    normalized_hgg    = hgg_utils.get_hgg_paths().parent.joinpath('scaled_hgg')
    all_patient_paths = hgg_utils.get_each_hgg_folder()

    print("Normalized data will be downloaded to following directory:")
    print(normalized_hgg)
    if input("Download Data? y/n : ") == 'n':
        return

    file_types      = ["flair", "t1", "t1ce", "t2", "seg"]
    file_extension  = ".nii.gz"
    if not normalized_hgg.exists():
        normalized_hgg.mkdir()

    for i in tqdm(range(len(all_patient_paths))):  
        patient_path           = all_patient_paths[i]                   # path to current patient
        patient_data           = hgg_utils.get_patient_data_at_index(i) # data for current patient
        normalized_data_path   = normalized_hgg.joinpath(patient_path.name)
        normalize_patient_data = scale_volume(patient_data, scale_value)
        if not normalized_data_path.exists():
            normalized_data_path.mkdir()
        
        for modality in file_types:
            file_path = normalized_data_path.joinpath("{}{}{}{}".format(patient_path.name, '_', modality, file_extension))
            temp = [slice_group[modality] for slice_group in normalize_patient_data]
            array_data = np.asarray(temp)
            array_data = np.moveaxis(array_data, 0, -1)
            affine = np.diag([1, 1, 1, 1]) 
            array_img  = nib.Nifti1Image(array_data, affine)
            nib.save(array_img, file_path)