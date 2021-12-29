import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm
import random
import pickle 
import scipy.ndimage as nd 
import scipy 
import math
from PIL import Image
import sys 
import os 
from resnet import resblock as residual_block
from resnet import relu
from resnet import batch_norm as batch_norm_resnet  
from resnet import resblock
from resnet import bottle_resblock
from resnet import conv
import tf_common_layer as common


def resnet18unet(x, is_training, ch_in = 3, ch_out = 2, ch = 64):
    x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
    x1 = x 
    features = resnet_template(x, is_training = is_training, res_n = 18, ch = ch)
    features = [features[i] for i in range(len(features))]
    enc_dim = [ch, ch*2, ch*4, ch*8]
    # 4x 8x 16x 32x

    features[3], _, _ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training)
    features[3], _, _ = common.create_conv_layer('mid_2', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training)
    features[3], _, _ = common.create_conv_layer('mid_3', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training)



    def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu"):
        x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm, deconv=True)
        
        x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

        x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
        x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

        return x 
    
    x = aggregate_block(features[2], features[3], enc_dim[2], enc_dim[3], enc_dim[2], "agg1", batchnorm=True)
    x = aggregate_block(features[1], x, enc_dim[1], enc_dim[2], enc_dim[1], "agg2", batchnorm=True)
    x = aggregate_block(features[0], x, enc_dim[0], enc_dim[1], enc_dim[0], "agg3", batchnorm=True)
    x = aggregate_block(x1, x, enc_dim[0], enc_dim[0], ch, "agg4", batchnorm=True)

    x, _, _ = common.create_conv_layer('agg5', x, ch, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True)
    x, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, deconv=False, activation = "linear")
    

    return x 

def resnet18unetNo2xZoom(x, is_training, ch_in = 3, ch_out = 2, ch = 64, noskip = False):
    x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False)
    x1 = x 
    features = resnet_template(x, is_training = is_training, res_n = 18, ch = ch)
    features = [features[i] for i in range(len(features))]
    enc_dim = [ch, ch*2, ch*4, ch*8]
    # 2x 4x 8x 16x

    features[3], _, _ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training,batchnorm = True)
    features[3], _, _ = common.create_conv_layer('mid_2', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training,batchnorm = True)
    features[3], _, _ = common.create_conv_layer('mid_3', features[3], enc_dim[3], enc_dim[3], kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training,batchnorm = True)



    def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu"):
        x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm, deconv=True)
        
        if noskip:
            x = x2
            x, _, _ = common.create_conv_layer(name+"_2", x, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
            x, _, _ = common.create_conv_layer(name+"_3", x, in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)
        else:
            x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2
            
            x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
            x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

        return x 
    
    x = aggregate_block(features[2], features[3], enc_dim[2], enc_dim[3], enc_dim[2], "agg1", batchnorm=True)
    x = aggregate_block(features[1], x, enc_dim[1], enc_dim[2], enc_dim[1], "agg2", batchnorm=True)
    x = aggregate_block(features[0], x, enc_dim[0], enc_dim[1], enc_dim[0], "agg3", batchnorm=True)
    x = aggregate_block(x1, x, enc_dim[0], enc_dim[0], ch, "agg4", batchnorm=True)

    #x, _, _ = common.create_conv_layer('agg5', x, ch, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True)
    x, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, deconv=False, activation = "linear")
    

    return x 


def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x


def resnet_template(x, is_training=True, reuse=False, res_n = 18, ch = 64, feature_activation = tf.nn.relu):
    with tf.variable_scope("resnet", reuse=reuse):
        
        residual_block = resblock
        if res_n < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        #ch = 8 # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=2, scope='conv')

        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        f1 = feature_activation(batch_norm_resnet(x, is_training, scope='batch_norm_f1')) 
        ########################################################################################################

        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        f2 = feature_activation(batch_norm_resnet(x, is_training, scope='batch_norm_f2')) 
        ########################################################################################################

        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

        f3 = feature_activation(batch_norm_resnet(x, is_training, scope='batch_norm_f3')) 
        ########################################################################################################

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

        f4 = feature_activation(batch_norm_resnet(x, is_training, scope='batch_norm_f4')) 
        ########################################################################################################

        # x = batch_norm(x, is_training, scope='batch_norm')
        # x = relu(x)

        # x = global_avg_pooling(x)
        # x = fully_conneted(x, units=feature_size, scope='logit')

        return f1,f2,f3,f4 

