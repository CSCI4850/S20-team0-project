

import pathlib

import nibabel as nib
import matplotlib.pyplot as plt

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
    grouped_slices = []

    # get paths to all patient volumes, or return empty list if paths not correct
    all_volume_paths = get_scans_at_index(i)

    # initialize data list
    for i in range(155):
        v_slice = {"flair": None, "t1": None, "t1ce": None, "t2": None, "seg": None, "s_id": i}
        grouped_slices.append(v_slice)

    for vol_path in all_volume_paths:
        modality_type = [m_type for m_type in file_types if vol_path.match("*{}{}".format(m_type, file_extensions))][0]
        volume_data    = nib.load(str(vol_path)).get_data()
        
        # add each slice of brain volume to volume_data list
        for slice_num in range(155):
            current_slice = volume_data[:,:,slice_num]
            if grouped_slices[slice_num]["s_id"] == slice_num:
                grouped_slices[slice_num][modality_type] = current_slice

    return grouped_slices

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