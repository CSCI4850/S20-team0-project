from keras import backend as K

# Much of this came from https://github.com/jocicmarko/ultrasound-nerve-segmentation

def dice_coef(truth, prediction, smooth=1):
    true_f = K.flatten(truth)
    pred_f = K.flatten(prediction)
    intersection = K.sum(true_f * pred_f)
    numerator = (2. * intersection) + smooth
    denominator = K.sum(true_f) + K.sum(pred_f) + smooth
    return numerator / denominator

def dice_loss(truth, prediction):
    return 1 - dice_coef(truth, prediction)

