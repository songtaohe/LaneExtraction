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
import tensorflow.contrib as tf_contrib

def resnet34unet(x, is_training, ch_in = 3, ch_out = 2, ch = 64):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = 34, ch = ch)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 2x 4x 8x 16x

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

	x, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True, activation = "linear")
	

	return x 

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
	with tf.variable_scope(scope):
		#x = flatten(x)
		x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

		return x


def resnet34unet_v3(x, is_training, ch_in = 3, ch_out = 2, ch = 64, feature_out = False, global_feature_func = None, res_n = 34):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch*2, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = res_n, ch = ch)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 2x 4x 8x 16x

	def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu"):
		x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm, deconv=True)
		
		x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

		x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
		x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

		return x 

	# wider receptive field 
	features[3], _, _ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=2)
	features[3], _, _ = common.create_conv_layer('mid_2', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=4)
	features[3], _, _ = common.create_conv_layer('mid_3', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=4)
	features[3], _, _ = common.create_conv_layer('mid_4', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=2)
	features[3], _, _ = common.create_conv_layer('mid_5', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=1)

	if global_feature_func is not None:
		features[3] = global_feature_func(features[3])

	if feature_out:
		f_out, _, _ = common.create_conv_layer('f_1', features[3], enc_dim[3], ch * 4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm=True) # 32x 
		f_out, _, _ = common.create_conv_layer('f_2', f_out, ch*4, ch * 2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x
		f_out, _, _ = common.create_conv_layer('f_3', f_out, ch*2, ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x

		f_out = tf.reduce_mean(f_out, axis = [1,2])
		f_out = fully_conneted(f_out, 2)
		 

	def prediction_conv(x, ch_in, ch_out, name):
		x, _, _ = common.create_conv_layer(name+"_1", x, ch_in, ch_out, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		#x, _, _ = common.create_conv_layer(name+"_2", x, ch_in, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		return x 

	x = aggregate_block(features[2], features[3], enc_dim[2], enc_dim[3], enc_dim[2], "agg1", batchnorm=True)
	#p16 = prediction_conv(x, enc_dim[2], ch_out, "pred16")
	x = aggregate_block(features[1], x, enc_dim[1], enc_dim[2], enc_dim[1], "agg2", batchnorm=True)
	#p8 = prediction_conv(x, enc_dim[1], ch_out, "pred8")
	x = aggregate_block(features[0], x, enc_dim[0], enc_dim[1], enc_dim[0], "agg3", batchnorm=True)
	#p4 = prediction_conv(x, enc_dim[0], ch_out, "pred4")
	x = aggregate_block(x1, x, ch*2, enc_dim[0], ch, "agg4", batchnorm=True)
	#p2 = prediction_conv(x, ch, ch_out, "pred2")
	p1, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True, activation = "linear")
	
	if feature_out:
		return p1,f_out
	else:
		return p1


def unet(x, is_training, ch_in = 3, ch_out = 2, ch = 64, feature_out = False, global_feature_func = None, res_n = 34):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch*2, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
	x1 = x 
	features = [] 
	
	x, _, _ = common.create_conv_layer("enc_2", x, ch * 2, ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_3", x, ch, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_4", x, ch*2, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_5", x, ch*2, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_6", x, ch*4, ch*4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_7", x, ch*4, ch*8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_8", x, ch*8, ch*8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)
		
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 2x 4x 8x 16x

	def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu"):
		x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm, deconv=True)
		
		x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

		x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
		x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

		return x 

	if global_feature_func is not None:
		features[3] = global_feature_func(features[3])

	if feature_out:
		f_out, _, _ = common.create_conv_layer('f_1', features[3], enc_dim[3], ch * 4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm=True) # 32x 
		f_out, _, _ = common.create_conv_layer('f_2', f_out, ch*4, ch * 2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x
		f_out, _, _ = common.create_conv_layer('f_3', f_out, ch*2, ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x

		f_out = tf.reduce_mean(f_out, axis = [1,2])
		f_out = fully_conneted(f_out, 2)
		 

	def prediction_conv(x, ch_in, ch_out, name):
		x, _, _ = common.create_conv_layer(name+"_1", x, ch_in, ch_out, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		#x, _, _ = common.create_conv_layer(name+"_2", x, ch_in, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		return x 

	x = aggregate_block(features[2], features[3], enc_dim[2], enc_dim[3], enc_dim[2], "agg1", batchnorm=True)
	#p16 = prediction_conv(x, enc_dim[2], ch_out, "pred16")
	x = aggregate_block(features[1], x, enc_dim[1], enc_dim[2], enc_dim[1], "agg2", batchnorm=True)
	#p8 = prediction_conv(x, enc_dim[1], ch_out, "pred8")
	x = aggregate_block(features[0], x, enc_dim[0], enc_dim[1], enc_dim[0], "agg3", batchnorm=True)
	#p4 = prediction_conv(x, enc_dim[0], ch_out, "pred4")
	x = aggregate_block(x1, x, ch*2, enc_dim[0], ch, "agg4", batchnorm=True)
	#p2 = prediction_conv(x, ch, ch_out, "pred2")
	p1, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True, activation = "linear")
	
	if feature_out:
		return p1,f_out
	else:
		return p1

def unet_dilated(x, is_training, ch_in = 3, ch_out = 2, ch = 64, feature_out = False, global_feature_func = None, res_n = 34):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch*2, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
	x1 = x 
	features = [] 
	
	x, _, _ = common.create_conv_layer("enc_2", x, ch * 2, ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_3", x, ch, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_4", x, ch*2, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_5", x, ch*2, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_6", x, ch*4, ch*4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)

	x, _, _ = common.create_conv_layer("enc_7", x, ch*4, ch*8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, deconv=False)
	x, _, _ = common.create_conv_layer("enc_8", x, ch*8, ch*8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True, deconv=False)
	features.append(x)
		
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 2x 4x 8x 16x

	features[3], _, _ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=2)
	features[3], _, _ = common.create_conv_layer('mid_2', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=4)
	features[3], _, _ = common.create_conv_layer('mid_3', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=4)
	features[3], _, _ = common.create_conv_layer('mid_4', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=2)
	features[3], _, _ = common.create_conv_layer('mid_5', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True, dilation=1)



	def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu"):
		x2, _, _ = common.create_conv_layer(name+"_1", x2, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = batchnorm, deconv=True)
		
		x = tf.concat([x1,x2], axis=3) # in_ch1 + in_ch2

		x, _, _ = common.create_conv_layer(name+"_2", x, in_ch1 + in_ch2, in_ch1 + in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
		x, _, _ = common.create_conv_layer(name+"_3", x, in_ch1 + in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

		return x 

	if global_feature_func is not None:
		features[3] = global_feature_func(features[3])

	if feature_out:
		f_out, _, _ = common.create_conv_layer('f_1', features[3], enc_dim[3], ch * 4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm=True) # 32x 
		f_out, _, _ = common.create_conv_layer('f_2', f_out, ch*4, ch * 2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x
		f_out, _, _ = common.create_conv_layer('f_3', f_out, ch*2, ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm=True) # 64x

		f_out = tf.reduce_mean(f_out, axis = [1,2])
		f_out = fully_conneted(f_out, 2)
		 

	def prediction_conv(x, ch_in, ch_out, name):
		x, _, _ = common.create_conv_layer(name+"_1", x, ch_in, ch_out, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		#x, _, _ = common.create_conv_layer(name+"_2", x, ch_in, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		return x 

	x = aggregate_block(features[2], features[3], enc_dim[2], enc_dim[3], enc_dim[2], "agg1", batchnorm=True)
	#p16 = prediction_conv(x, enc_dim[2], ch_out, "pred16")
	x = aggregate_block(features[1], x, enc_dim[1], enc_dim[2], enc_dim[1], "agg2", batchnorm=True)
	#p8 = prediction_conv(x, enc_dim[1], ch_out, "pred8")
	x = aggregate_block(features[0], x, enc_dim[0], enc_dim[1], enc_dim[0], "agg3", batchnorm=True)
	#p4 = prediction_conv(x, enc_dim[0], ch_out, "pred4")
	x = aggregate_block(x1, x, ch*2, enc_dim[0], ch, "agg4", batchnorm=True)
	#p2 = prediction_conv(x, ch, ch_out, "pred2")
	p1, _, _ = common.create_conv_layer('output', x, ch, ch_out, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True, activation = "linear")
	
	if feature_out:
		return p1,f_out
	else:
		return p1




def resnet34FPN(x, is_training, input_dim, ch_in = 3, ch_out = 2, ch = 64, FPN_channel = 256):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = 34, ch = ch, feature_activation = lambda x: x)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 4x 8x 16x 32x
	

	def aggregate_block(x1, x2, in_ch1, in_ch2, out_ch, name, batchnorm=True, activation = "relu", scale=1):
		x2 = tf.image.resize(x2, [input_dim//scale, input_dim//scale], method='nearest')
		x1, _, _ = common.create_conv_layer(name+"_1", x1, in_ch1, in_ch2, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = "linear")
		
		x = x1 + x2 # in_ch2

		x, _, _ = common.create_conv_layer(name+"_2", x, in_ch2, in_ch2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm)
		x, _, _ = common.create_conv_layer(name+"_3", x, in_ch2, out_ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = batchnorm, activation = activation)

		return x 

	

	x32, _, _ = common.create_conv_layer("turningpoint_conv", features[3], enc_dim[3], FPN_channel, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True, activation = "linear")

	x16 = aggregate_block(features[2], x32, enc_dim[2], FPN_channel, FPN_channel, "agg1", batchnorm=True, scale = 16,  activation = "linear")
	x8 = aggregate_block(features[1], x16, enc_dim[1], FPN_channel, FPN_channel//2, "agg2", batchnorm=True, scale = 8,  activation = "linear")
	x4 = aggregate_block(features[0], x8, enc_dim[0], FPN_channel//2, FPN_channel//4, "agg3", batchnorm=True, scale = 4,  activation = "linear")
	x2 = aggregate_block(x1, x4, FPN_channel//4, FPN_channel//4, ch, "agg4", batchnorm=True, scale = 2,  activation = "linear")
	
	p1, _, _ = common.create_conv_layer('agg5', x2, ch, ch_out, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, deconv=True, activation = "linear")

	def prediction_conv(x, ch_in, ch_out, name):
		x, _, _ = common.create_conv_layer(name+"_1", x, ch_in, ch_in, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer(name+"_2", x, ch_in, ch_out, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = False, activation = "linear")
		return x 

	#p1 = prediction_conv(x1, ch, ch_out, "pred1")
	p2 = prediction_conv(x2, ch, ch_out, "pred2")
	p4 = prediction_conv(x4, FPN_channel//4, ch_out, "pred4")
	p8 = prediction_conv(x8, FPN_channel//2, ch_out, "pred8")
	p16 = prediction_conv(x16, FPN_channel, ch_out, "pred16")
	

	return p1,p2,p4,p8,p16



	# features[0] = tf.nn.max_pool(features[0], 3, 2, padding="SAME")
	# for i in range(2,4):
	#     features[i] = tf.image.resize(features[i], [self.input_dim//4, self.input_dim//4])
		
	# f = tf.concat(features, axis=3) # 3840 ... or 960

	# if res_n > 50:
	#     self.feature_ch = 960*4 //self.lite
	# else:
	#     self.feature_ch = 960 //self.lite






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

