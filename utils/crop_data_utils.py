"""
Utilities to be used in cropping images
"""

import utils.hgg_utils as hu
import numpy as np
import nibabel as nib
import pathlib
import tqdm as tqdm

"""
LAYERS_TO_CROP refers to the number of outer layers of pixels to be cropped from the image.
For example if layers_to_crop is set to 2:
Input (6x6):        Output (2x2, - denotes cropped pixel): 
123456                  ------
123456                  ------
123456                  --34--
123456                  --34--
123456                  ------
123456                  ------
"""
LAYERS_TO_CROP = 16

def crop_slice(slice):
    #####
    cropped_slice = slice[LAYERS_TO_CROP : -LAYERS_TO_CROP, LAYERS_TO_CROP : -LAYERS_TO_CROP]

def crop_patient_data(data):
    #Function to crop the data. input is all modalities and slices for a patient
    cropped_data = []
    for slice_group in data:
        v_slice = {"flair": crop_slice(slice_group["flair"]),
                   "t1"   : crop_slice(slice_group["t1"]),
                   "t1ce" : crop_slice(slice_group["t1ce"]),
                   "t2"   : crop_slice(slice_group["t2"]),
                   "seg"  : crop_slice(slice_group["seg"]),
                   "s_id" : slice_group["s_id"] }
        cropped_data.append(v_slice)
    return cropped_data


def crop_dataset_images():
    cropped_hgg_directory = hu.get_hgg_paths().parent.joinpath('cropped_hgg')
    all_patient_paths = hu.get_each_hgg_folder()

    print("Cropped slices will be saved in directory: ")
    print(cropped_hgg_directory)
    if ("Do you want to continue? y/n: ") == 'n':
        return

    file_types = ["flair", "t1", "t1ce", "t2", "seg"]
    file_extension = "nii.gz"

    # Check to see if directory folder already exists
    # before creating one.
    if not cropped_hgg_directory.exists():
        cropped_hgg_directory.mkdir()

    for i in tqdm(range(len(all_patient_paths))):
        patient_path = all_patient_paths[i]
        patient_data = hu.get_patient_data_at_index(i)
        cropped_data_path = cropped_hgg_directory.joinpath(patient_path.name)
        cropped_patient_data_ = crop_patient_data(patient_data)

        if not cropped_data_path.exists():
            cropped_data_path.mkdir()


