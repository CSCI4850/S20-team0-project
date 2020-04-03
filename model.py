import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

# Model code borrowed and adapted from: https://github.com/zhixuhao/unet
# Inputs need to be divisible? or power? of 2 -- due to correct sizing with concatentation and 2x2 downsampling and 2x2 upsampling

def unet(input_size, ds=1):
    '''
    ds: int, some multiple of 2, cuts number of filters uniformly
            across entire network for quick prototype of Unet for
            GPU RAM purposes
    '''
    inp = Input(input_size) 
    conv_0 = Conv2D(filters = 64//ds, 
                   kernel_size = 3, 
                   activation = 'relu', 
                   padding = 'same')(inp)
    conv_0 = Conv2D(filters = 64//ds, 
                   kernel_size = 3, 
                   activation = 'relu', 
                   padding = 'same')(conv_0)
    
    downsample_0 = MaxPooling2D(pool_size=2)(conv_0)
    
    conv_1 = Conv2D(filters = 128//ds, 
                   kernel_size = 3, 
                   activation = 'relu', 
                   padding = 'same')(downsample_0)
    conv_1 = Conv2D(filters = 128//ds, 
                   kernel_size = 3, 
                   activation = 'relu', 
                   padding = 'same')(conv_1)
    
    downsample_1 = MaxPooling2D(pool_size=2)(conv_1)
    
    
    conv_2 = Conv2D(filters = 256//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(downsample_1)
    conv_2 = Conv2D(filters = 256//ds, 
                   kernel_size = 3, 
                   activation = 'relu', 
                   padding = 'same')(conv_2)
    
    downsample_2 = MaxPooling2D(pool_size=2)(conv_2)
    
    conv_3 = Conv2D(filters = 512//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(downsample_2)
    conv_3 = Conv2D(filters = 512//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_3)

    downsample_3 = MaxPooling2D(pool_size=2)(conv_3)

    conv_4 = Conv2D(filters = 1024//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(downsample_3)
    conv_4 = Conv2D(filters = 1024//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_4)

    upsample_5 = Conv2D(filters=512//ds, 
                        kernel_size=2, 
                        activation = 'relu', 
                        padding = 'same')(UpSampling2D(size = 2)(conv_4))
    merge_5 = concatenate([conv_3, upsample_5], axis = 3)
    conv_5 = Conv2D(filters = 512//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(merge_5)
    conv_5 = Conv2D(filters = 512//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_5)

    upsample_6 = Conv2D(filters = 256//ds, 
                        kernel_size = 2, 
                        activation = 'relu', 
                        padding = 'same')(UpSampling2D(size = 2)(conv_5))
    merge_6 = concatenate([conv_2, upsample_6], axis = 3)
    conv_6 = Conv2D(filters = 256//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(merge_6)
    conv_6 = Conv2D(filters = 256//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_6)

    upsample_7 = Conv2D(filters = 128//ds, 
                        kernel_size = 2, 
                        activation = 'relu', 
                        padding = 'same')(UpSampling2D(size = 2)(conv_6))
    merge_7 = concatenate([conv_1, upsample_7], axis = 3)
    conv_7 = Conv2D(filters = 128//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(merge_7)
    conv_7 = Conv2D(filters = 128//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_7)

    upsample_8 = Conv2D(filters = 64//ds, 
                        kernel_size = 2, 
                        activation = 'relu', 
                        padding = 'same')(UpSampling2D(size = 2)(conv_7))
    merge_8 = concatenate([conv_0, upsample_8], axis = 3)
    conv_8 = Conv2D(filters = 64//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(merge_8)
    conv_8 = Conv2D(filters = 64//ds, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_8)
    conv_8 = Conv2D(filters = 2, 
                    kernel_size = 3, 
                    activation = 'relu', 
                    padding = 'same')(conv_8)
    
    conv_9 = Conv2D(filters = 1, 
                    kernel_size = 1, 
                    activation = 'sigmoid')(conv_8)
    
    
    model = Model(inputs=inp, outputs=conv_9)

    return model