from utils import hgg_utils
import numpy as np
import pandas as pd

def 

def normalize_brain_volume(brain_volume):
        '''
        - brain volume shape (240,240,155)
        - 155 slices, 240x240 pixels each
        - this function returns a normalized brain volume as a numpy array 
        '''
        normalized_brain_volume = np.empty([240,240,155])
        
        # iterate through each slice in brain volume",
        for slice_num in range(154):
            current_slice            = brain_volume[:,:,slice_num]
            brain_region_intensities = get_brain_region(current_slice)
            intensity_mean           = np.mean(brain_region_intensities)
            intensity_standard_dev   = np.std(brain_region_intensities)
            
            # fill apply normalization algo to non-zero values and fill normalized_brain_volume array",
            for i in range(239):
                for j in range(239):
                    if current_slice[i,j] != 0:
                        normalized_brain_volume[i,j,slice_num] = (current_slice[i,j] - intensity_mean) / intensity_standard_dev
                    else:
                        normalized_brain_volume[i,j,slice_num] = 0
                        
        return normalized_brain_volume


if __name__ == "__main__":
    volume_data = hgg_utils.get_patient_data_at_index(0)
    hgg_utils.display_slice(volume_data, 75, 'flair')

    normalize_data = normalize_brain_volume(volume_data)