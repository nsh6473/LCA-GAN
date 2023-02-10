# from turtle import down
import tensorflow as tf

import numpy as np

# regularizers = tf.keras.regularizers.l2(0.001)
regularizers = tf.keras.regularizers.l2(0.02)

# Channel attention block
def channel_attention_block(input, filters, kernel_size=3, reduction=4, use_bias=False, down=False, up=False):
    # define skip and resize input feature if resize is True
    if down==True:
        skip = tf.image.resize(input, (int(input.shape[1]//2), int(input.shape[2]//2)))
        conv1   = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=4,
                                         strides=2,
                                         kernel_regularizer=regularizers,
                                         padding='same', use_bias=use_bias)(input)
    elif up==True:
        skip = tf.image.resize(input, (int(input.shape[1]*2), int(input.shape[2]*2)))
        conv1   = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                  kernel_size=4,
                                                  strides=2,
                                                  kernel_regularizer=regularizers,
                                                  padding='same', use_bias=use_bias)(input)
    else:
        skip = input
        conv1   = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        kernel_regularizer=regularizers,
                                        padding='same', use_bias=use_bias)(input)

    # extract feature for attention
    relu    = tf.keras.layers.ReLU()(conv1)
    conv2   = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_regularizer=regularizers,
                                     padding='same', use_bias=use_bias)(relu)   # conv2 = batch, n/2, n/2, filters
    
    # extract spatial attention
    gap     = tf.keras.layers.GlobalAveragePooling2D()(conv2)                   # gap = batch, 1, 1, filters
    exp_dim1= tf.expand_dims(gap, 1)
    exp_dim2= tf.expand_dims(exp_dim1, 1)

    mlp1    = tf.keras.layers.Conv2D(filters=filters//reduction,
                                     kernel_size=1,
                                     kernel_regularizer=regularizers,
                                     use_bias=use_bias)(exp_dim2)               # mlp1 = batch, 1, 1, filters//reduction
    relu2   = tf.keras.layers.ReLU()(mlp1)
    mlp2    = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=1,
                                     kernel_regularizer=regularizers,
                                     use_bias=use_bias)(relu2)                  # mlp2 = batch, 1, 1, filters
    sigmoid = tf.nn.sigmoid(mlp2)

    # adjust channel attention about input
    mul     = sigmoid * conv2                                                   # mul = batch, n/2, n/2, filters
    add     = mul + skip                                                        # add = batch, n/2, n/2, filters
    return add

# Spatial attention block
def spatial_attention_block(input, filters, kernel_size=3, use_bias=False, resize=False):
    if resize==True:
        skip = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', kernel_regularizer=regularizers, use_bias=use_bias)(input)
    else:
        skip = input
    # extract feature for attention
    conv1   = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_regularizer=regularizers,
                                     padding='same', use_bias=use_bias)(input)
    conv2   = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_regularizer=regularizers,
                                     padding='same', use_bias=use_bias)(conv1)
    
    # extract spatial attention
    cap     = tf.reduce_mean(conv2, axis=[3], keepdims=True)
    conv3   = tf.keras.layers.Conv2D(filters=1,
                                     kernel_size=7,
                                     kernel_regularizer=regularizers,
                                     padding='same', use_bias=use_bias)(cap)
    sigmoid = tf.nn.sigmoid(conv3)

    # adjust channel attention about input
    mul     = conv2 * sigmoid
    add     = mul + skip
    return add

# Unet
def Unet(input_shape=(256,256,3), kernel_size=3, reduction=4, use_bias=False):
    # batch, 256, 256, 3
    inputs = tf.keras.Input(input_shape)
    # batch, 256, 256, 64
    conv = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding='same', kernel_regularizer=regularizers, use_bias=use_bias)(inputs)
    sab = spatial_attention_block(conv, filters=64)

    # Encoder----------------------------------------------------------------------------------------------------
    # Encoder1  256, 256, 64    128, 128, 64    128, 128, 128
    down_cab1 = channel_attention_block(sab, filters=64, down=True)
    down_sab1 = spatial_attention_block(down_cab1, filters=128, resize=True)

    # Encoder2  128, 128, 128   64,  64,  128   64, 64, 256 
    down_cab2 = channel_attention_block(down_sab1, filters=128, down=True)
    down_sab2 = spatial_attention_block(down_cab2, filters=256, resize=True)

    # Encoder3  64, 64, 256     32, 32, 256     32, 32, 512
    down_cab3 = channel_attention_block(down_sab2, filters=256, down=True)
    down_sab3 = spatial_attention_block(down_cab3, filters=512, resize=True)

    # Encoder4  32, 32, 512     16, 16, 512     16, 16, 512
    down_cab4 = channel_attention_block(down_sab3, filters=512, down=True)
    down_sab4 = spatial_attention_block(down_cab4, filters=512)

    # Encoder5  16, 16, 512     8, 8, 512       8, 8, 512
    down_cab5 = channel_attention_block(down_sab4, filters=512, down=True)
    down_sab5 = spatial_attention_block(down_cab5, filters=512)

    # Decoder----------------------------------------------------------------------------------------------------
    # Decoder1  8, 8, 512       16, 16, 512     16, 16, 1024    16, 16, 512
    up_cab1 = channel_attention_block(down_sab5, filters=512, up=True)
    up_conc1 = tf.keras.layers.Concatenate()([up_cab1, down_sab4])
    up_sab1 = spatial_attention_block(up_conc1, filters=512, resize=True)

    # Decoder2  16, 16, 512     32, 32, 512     32, 32, 1024    32, 32, 512
    up_cab2 = channel_attention_block(up_sab1, filters=512, up=True)
    up_conc2 = tf.keras.layers.Concatenate()([up_cab2, down_sab3])
    up_sab2 = spatial_attention_block(up_conc2, filters=256, resize=True)
    
    # Decoder3  32, 32, 512     64, 64, 512     64, 64, 512    64, 64, 256
    up_cab3 = channel_attention_block(up_sab2, filters=256, up=True)
    up_conc3 = tf.keras.layers.Concatenate()([up_cab3, down_sab2])
    up_sab3 = spatial_attention_block(up_conc3, filters=128, resize= True)

    # Decoder4  64, 64, 512     128, 128, 512   128, 128, 1024  128, 128, 512
    up_cab4 = channel_attention_block(up_sab3, filters=128, up=True)
    up_conc4 = tf.keras.layers.Concatenate()([up_cab4, down_sab1])
    up_sab4 = spatial_attention_block(up_conc4, filters=64, resize=True)
    
    # Decoder5  128, 128, 512   256, 256, 512   256, 256, 1024  256, 256, 512
    up_cab5 = channel_attention_block(up_sab4, filters=64, up=True)
    up_sab5 = spatial_attention_block(up_cab5, filters=64)

    # gen image--------------------------------------------------------------------------------------------------
    out = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', kernel_regularizer=regularizers, use_bias=use_bias)(up_sab5)

    return tf.keras.Model(inputs=inputs, outputs=out)

# Discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    conv1 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.LeakyReLU()(bn1)

    conv2 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.layers.LeakyReLU()(bn2)

    conv3 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(relu2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    relu3 = tf.keras.layers.LeakyReLU()(bn3)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(relu3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)