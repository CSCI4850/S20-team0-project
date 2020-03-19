

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
        
def get_multimodal_volume_for_patient(patient_path):
    
    """
    Purpose:
        Returns a multimodal volume 240x240x155x4 as a numpy array of 32 bit floats.
        
    Args:
        Path obj to a patient folder
    
    """
    
    non_mask_modalities_paths = [modality for modality in patient_path.iterdir() if not modality.match("*seg.nii.gz") ]
    
    non_mask_modalities_vols = [nib.load(x).get_fdata(dtype=np.float32) for x in non_mask_modalities_paths]

    multimodal_vol = np.stack(
                            (
                            non_mask_modalities_vols[0],
                            non_mask_modalities_vols[1],
                            non_mask_modalities_vols[2],
                            non_mask_modalities_vols[3]
                            ),

                            axis=3

    )
    
    return multimodal_vol

def get_all_input_volumes():

    """
    Purpose:
        -returns the X input tensor of shape (259, 240, 240, 155, 4) where:
            -dim 0   -> patient index
            -dim 1&2 -> slice dimension
            -dim3    -> slice index
            -dim4    -> modalities
    
    """
    
    all_patient_paths = get_each_hgg_folder()

    all_multimodal_volumes = np.empty([259, 240, 240, 155, 4], dtype=np.float32)
    
    for index, patient in tqdm( enumerate(all_patient_paths), total=len(all_patient_paths) ):
        all_multimodal_volumes[index, :, :, :, :] = get_multimodal_volume_for_patient(patient_path=patient)
        
    return all_multimodal_volumes



def print_multimodal_slice(multimod_vol, patient_idx=np.nan, slice_idx=0):

    """
    Purpose:
        Print a slice from a multimodal volume.
        This plots each 2D modality that is a part of the multimodal slice.
        Will print flair slice, t1 slice, t1ce slice, t2 slice.
        
    Args:
        multimodal_vol
            -numpy array of a multimodal volume. expected shape: (259, 240,240,155,4)
        
        patient_idx
            -index of patient
            
        slice_idx
            -slice index to be printed
    
    """
        
    # if multimodal vol is for 1 patient
    # shape (240, 240, 155, 4)
    if np.isnan(patient_idx):
        
        # multimodal_vol is actually a multimodal vol
        if multimod_vol.shape[-1] == 4:
            plt.figure(1, figsize=(8,8))

            plt.subplot(221)
            plt.title("flair")
            plt.imshow(multimod_vol[:,:,slice_idx,0].T, cmap="Greys_r")

            plt.subplot(222)
            plt.title("t1")
            plt.imshow(multimod_vol[:,:,slice_idx,1].T, cmap="Greys_r")

            plt.subplot(223)
            plt.title("t1ce")
            plt.imshow(multimod_vol[:,:,slice_idx,2].T, cmap="Greys_r")

            plt.subplot(224)
            plt.title("t2")
            plt.imshow(multimod_vol[:,:,slice_idx,3].T, cmap="Greys_r")

            plt.tight_layout()
            plt.show()
            
        # multimodal_vol is actually a mask -- this function should probably be renamed
        elif multimod_vol.shape[-1] == 1:
            plt.figure(1, figsize=(8,8))

            plt.subplot(221)
            plt.title("mask")
            plt.imshow(multimod_vol[:,:,slice_idx,0].T, cmap="Greys_r")

            plt.tight_layout()
            plt.show()
    
    # if multimodal vol is for all patients
    # shape (259, 240, 240, 155, 4)
    else:
        plt.figure(1, figsize=(8,8))

        plt.subplot(221)
        plt.title("flair")
        plt.imshow(multimod_vol[patient_idx,:,:,slice_idx,0].T, cmap="Greys_r")

        plt.subplot(222)
        plt.title("t1")
        plt.imshow(multimod_vol[patient_idx,:,:,slice_idx,1].T, cmap="Greys_r")

        plt.subplot(223)
        plt.title("t1ce")
        plt.imshow(multimod_vol[patient_idx,:,:,slice_idx,2].T, cmap="Greys_r")

        plt.subplot(224)
        plt.title("t2")
        plt.imshow(multimod_vol[patient_idx,:,:,slice_idx,3].T, cmap="Greys_r")

        plt.tight_layout()
        plt.show()
    
    
def convert_mask_to_binary_mask( mask ):

    """
    Purpose:
        Given a mask tensor, convert all non-zero values to 1.
        This converts the mask into a binary mask.
        
    Args: 
        mask
            -a mask tensor as a numpy array
    
    
    """
    
    # Any val greater than 0 becomes 1
    mask[np.where(mask > 0)] = 1
    
    return mask

def get_a_multimodal_tensor( patient_path ):

    """
    Purpose:
        Return a multimodal tensor of np.float32 for a patient path.
        ie. combine the flair volume, t1 volume, t1ce volume, and t2 volume for a patient
        shape will be (240, 240, 155, 4)
        idx 0 & 1 -> heaight and width
        idx 2 -> slices
        idx 3 -> channels (flair, t1, t1ce, t2)
    
    Args:
        patient_path
            -path obj to patient folder
    
    """
    
    
    multimodal_tensor = np.ones([240, 240, 155, 4], dtype=np.float32)

    four_modalities = [scan for scan in patient_path.iterdir() if not scan.match("*seg.nii.gz")]

    for idx, scan in enumerate(four_modalities):

        multimodal_tensor[:, :, :, idx] = nib.load(scan).get_fdata() 
    
    return multimodal_tensor


def get_a_mask_tensor( patient_path ):

    """
    Purpose:
        Return a mask tensor of np.float32 for a patient path.
        shape will be (240, 240, 155, 1)
        idx 0 & 1 -> heaight and width
        idx 2 -> slices
        idx 3 -> channels (flair, t1, t1ce, t2)
    
    Args:
        patient_path
            -path obj to patient folder
    
    """
    
    mask = np.ones([240, 240, 155, 1], dtype=np.float32)

    mask_path = [scan for scan in patient_path.iterdir() if scan.match("*seg.nii.gz")]

    for scan in mask_path:
        
        mask[:, :, :, 0] = nib.load(scan).get_fdata() 
    
    return mask



    
    