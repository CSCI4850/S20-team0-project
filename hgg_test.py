from utils import hgg_utils

if __name__ == "__main__":
    volume_data = hgg_utils.get_patient_data_at_index(0)
    hgg_utils.display_slice(volume_data, 75, 'flair')