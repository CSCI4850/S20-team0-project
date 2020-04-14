"""
Utilities to be used in cropping images
"""

import utils.hgg_utils as hu
import nibabel as nib
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

def save_cropped_data(tensor, affines_list, mod_paths, destination):
    patient_paths = [x.parent.stem for x in mod_paths]
    mods = [x.name for x in mod_paths]

    for modality in range(tensor.shape[-1]):

        new_file_name = "cropped_" + str(mods[modality])

        new_patient_folder = destination.joinpath(patient_paths[modality])

        if not new_patient_folder.exists():
            new_patient_folder.mkdir()

        new_dest = new_patient_folder.joinpath(new_file_name)

        a = nib.Nifti1Image(tensor[:, :, :, modality], affine=affines_list[modality])

        nib.save(a, new_dest)

def crop_patient_tensor(tensor):
    return tensor[LAYERS_TO_CROP : -LAYERS_TO_CROP, LAYERS_TO_CROP : -LAYERS_TO_CROP, :, :]


def crop_dataset_images():
    # Define name of folder to save data to
    cropped_hgg_directory = hu.get_hgg_paths().parent.joinpath('cropped_hgg')
    # Get paths to all patient folders
    all_patient_paths = hu.get_each_hgg_folder()

    # Print path to directory where data will be saved
    print("Cropped slices will be saved in directory: ")
    print(cropped_hgg_directory)

    # Check to see if directory folder already exists
    # before creating one.
    if not cropped_hgg_directory.exists():
        cropped_hgg_directory.mkdir()

    # Iterate through each patient
    #   Load patient tensor
    #   Crop tensor
    #   Save tensor
    for patient in tqdm(all_patient_paths):
        X, affines, paths = hu.get_a_multimodal_tensor(patient)
        cropped_tensor = crop_patient_tensor(X)
        save_cropped_data(cropped_tensor, affines, paths, cropped_hgg_directory)


