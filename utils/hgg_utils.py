

import pathlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def get_hgg_paths():
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    ./some_directory_to_hold_everything/
        +-- MICCAI_BraTS_2019_Data_Training
        |
        |   +-- MICCAI_BraTS_2019_Data_Training
        |
        |   |   + -- HGG
        |
        +-- S20-team0-project  */ Cloned github repo */
        
    Return path to HGG directory
    """
    
    for path in sorted ( list( pathlib.Path.cwd().parent.iterdir() ) ):
        if path.name == "MICCAI_BraTS_2019_Data_Training":
            hgg_folders_path =  sorted(
                                        list(
                                            sorted(
                                                    list(
                                                        path.iterdir()
                                                    )
                                            )[0].iterdir()
                                        )
                                )[0]

    return hgg_folders_path


def get_each_hgg_folder( ):
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    Returns a sorted list of paths to each HGG folder
    """
    
    return sorted( [folder for folder in get_hgg_paths().iterdir() ] )
    
def get_scans_at_index( i ):
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    Returns a sorted list of the 5 modalities inside hgg folder at index i
    """
    
    return sorted( [modality for modality in get_each_hgg_folder()[i].iterdir()] )
    
def get_patient_data_at_index(i):
    """
    Returns a list of dictionaries with each dictionary containing a single slice across all modalities
    """

    file_types      = ["flair", "t1", "t1ce", "t2", "seg"]
    file_extensions = ".nii.gz"
    grouped_slices  = []

    # get paths to all patient volumes, or return empty list if paths not correct
    all_volume_paths = get_scans_at_index(i)

    # initialize data list
    for i in range(155):
        v_slice = {"flair": None, "t1": None, "t1ce": None, "t2": None, "seg": None, "s_id": i}
        grouped_slices.append(v_slice)

    for vol_path in all_volume_paths:
        modality_type = [m_type for m_type in file_types if vol_path.match("*{}{}".format(m_type, file_extensions))][0]
        volume_data   = nib.load(str(vol_path)).get_data()
        
        # add each slice of brain volume to volume_data list
        for slice_num in range(155):
            current_slice = volume_data[:,:,slice_num]
            if grouped_slices[slice_num]["s_id"] == slice_num:
                grouped_slices[slice_num][modality_type] = current_slice

    return grouped_slices


def get_mask_volume_at_index( index ): 

    """
    Purpose:
        Return the mask/label volume correspoding to one patient at index 
    
    Args:
        index: 
            -index of patient whose mask volume you want to get
    """
    
    list_of_single_patient_slice_dictionaries = get_patient_data_at_index( index )
    
    mask_volume = []
    
    for mask_slice_index in range(len( list_of_single_patient_slice_dictionaries )):
        mask_volume.append( list_of_single_patient_slice_dictionaries[mask_slice_index]["seg"] )
        
    return mask_volume

def display_slice(slice_list, slice_num, image_type):
    """
    Displays an image of desired slice
    """
    if slice_num < 0:
        print("Index out of range")
        return
    if len(slice_list) == 0:
        print("No data is loaded")
        return
    if slice_num < len(slice_list):
        # .T corrects img orientation, Greys_r gives correct black & white img
        plt.imshow(slice_list[slice_num][image_type].T, cmap='Greys_r') 
        plt.show()
        
def get_a_mask(patient_path):
    
    """
    Purpose:
        Returns a mask volume 240x240x155 as a numpy array of 32 bit floats.
        
    Args:
        Path obj to a patient folder
    
    """
    
    # path to mask volume for a patient folder
    mask_volume_path = [modality for modality in patient_path.iterdir() if modality.match("*seg.nii.gz") ][0]
    
    # load mask vol into memory
    mask_vol = nib.load(mask_volume_path).get_fdata(dtype=np.float32)
    
    return mask_vol # of shape 240x240x155

def get_all_mask_volumes():
    
    """
    Purpose:
        Returns all masks for dataset in a single numpy array of 32 bit floats.
        Shape will be 259x240x240x155x1
    """
    
    all_patient_paths = get_each_hgg_folder()
    
    # preallocate empty np array to hold all the masks
    # dim 0   -> each patient
    # dim 1&2 -> slice dimensions
    # dim 3   -> number of slices for patient
    # dim 4   -> necessary for feeding into CNN
    all_masks = np.empty([259, 240, 240, 155, 1], dtype=np.float32)
        
    
    # populate all_masks with every mask volume in dataset
    for index, patient_path in tqdm( enumerate(all_patient_paths), total=len(all_patient_paths)   ):
        
        all_masks[index, :, :, :, 0] = get_a_mask(patient_path)
    
    return all_masks
        
        