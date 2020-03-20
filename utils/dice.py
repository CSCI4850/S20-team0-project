# We need this to find the intersection of the two images:
def find_intersection(truth, prediction):
    # Args: True mask, predicted mask
    # Returns: The number of shared elements
    overlap = 0
    for i in range(0, truth.shape[0]):
        for j in range(0, truth.shape[1]):
            if truth[i,j] == prediction[i,j]:
                overlap += 1
    return overlap


def dice_loss(truth, prediction):
    # Dice coefficient score = (2 * (number of intersected pixels in truth and prediction) + 1)/ (total number of pixels (2 * 240 * 240) + 1)
    # The reason 1 is added to the numerator and denominator is to prevent division by zero without losing the ratio
        # You will often see this labeled as "smooth"
    overlap = find_intersection(truth,prediction)
    numerator = 2 * overlap + 1
    truth_num = truth.shape[0] * truth.shape[1]
    pred_num = prediction.shape[0] * prediction.shape[1]
    denominator = truth_num + pred_num + 1
    # Dice loss is just 1 - dice coefficient score
    return 1 - numerator/denominator