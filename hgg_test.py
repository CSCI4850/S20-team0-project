from utils import hgg_utils
from utils import normalize_data_utils as norm
import numpy as np
import nibabel as nib
import pathlib



def normalize_data_set():
    normalized_hgg = hgg_utils.get_hgg_paths().parent.joinpath('normalized_hgg')
    all_patient_paths = hgg_utils.get_each_hgg_folder()

    file_types      = ["flair", "t1", "t1ce", "t2", "seg"]
    file_extension  = ".nii.gz"
    test_path = pathlib.Path('/home/josh/Desktop/py/envs')

    for i in range(1):  #len(all_patient_paths)):
        patient_path         = all_patient_paths[i]                   # path to current patient
        print(patient_path)
        patient_data         = hgg_utils.get_patient_data_at_index(i) # data for current patient
        normalized_data_path = normalized_hgg.joinpath(patient_path.name)

        for modality in file_types:
            file_path = test_path.joinpath("{}{}{}{}".format(patient_path.name, '_', modality, file_extension))
            temp = [slice_group[modality] for slice_group in patient_data]
            array_data = np.asarray(temp)
            affine = np.diag([1, 2, 3, 1]) 
            array_img  = nib.Nifti1Image(array_data, affine) # look up: why does it need affine?
            #nib.save(array_img, file_path)
            print(file_path)

if __name__ == "__main__":
    volume_data = hgg_utils.get_patient_data_at_index(0)
    hgg_utils.display_slice(volume_data, 75, 'flair')

    norm_data = norm.mean_standard_norm_volume(volume_data)
    hgg_utils.display_slice(norm_data, 75, 'flair')
    

    