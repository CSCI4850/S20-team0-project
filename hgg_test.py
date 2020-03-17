from utils import hgg_utils
from utils import normalize_data_utils as norm


if __name__ == "__main__":
    volume_data = hgg_utils.get_patient_data_at_index(0)
    hgg_utils.display_slice(volume_data, 75, 'flair')

    normalized_data = norm.mean_standard_norm_volume(volume_data)
    print(normalized_data)
    hgg_utils.display_slice(normalized_data, 75, 'flair')

