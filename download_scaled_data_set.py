from utils import normalize_data_utils as norm

if __name__ == "__main__":
    '''
    Run this program to create normalized dataset
    Will download to directory : Creates a standardized dataset in dir : .../MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/scaled_hgg
    '''

    norm.download_scaled_dataset()