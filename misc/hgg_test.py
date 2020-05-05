from utils import hgg_utils
from utils import normalize_data_utils as norm
import numpy as np
import nibabel as nib
import pathlib
from tqdm import tqdm

        
if __name__ == "__main__":
    volume_data = hgg_utils.get_patient_data_at_index(0)
    hgg_utils.display_slice(volume_data, 75, 'flair')

    norm_data = norm.mean_standard_norm_volume(volume_data)
    hgg_utils.display_slice(norm_data, 75, 'flair')
    

    