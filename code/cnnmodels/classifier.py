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

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
	with tf.variable_scope(scope):
		#x = flatten(x)
		x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

		return x

def resnet18classifier(x, is_training, ch_in = 3, ch_out = 2, ch = 64, embedding_output = False, feature3_output = False):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, activation = "linear")
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = 18, ch = ch)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 4x 8x 16x 32x 


	g,_,_ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	g,_,_ = common.create_conv_layer('mid_2', g, enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	
	g = tf.reduce_mean(g, axis = [1,2])

	f = g
	f = fully_conneted(f, ch * 8, scope = "fc1")
	f = tf.nn.relu(f)
	
	if embedding_output:
		if feature3_output:
			return f, features[3]
		else:
			return f

	return fully_conneted(f, ch_out, scope = "fc2")

def resnet34classifier(x, is_training, ch_in = 3, ch_out = 2, ch = 64, embedding_output = False, feature3_output = False):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, activation = "linear")
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = 34, ch = ch)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch, ch*2, ch*4, ch*8]
	# 4x 8x 16x 32x 

	g,_,_ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	g,_,_ = common.create_conv_layer('mid_2', g, enc_dim[3], enc_dim[3], kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	g = tf.reduce_mean(g, axis = [1,2])

	f = g
	f = fully_conneted(f, ch * 8, scope = "fc1")
	f = tf.nn.relu(f)
	
	if embedding_output:
		if feature3_output:
			return f, features[3]
		else:
			return f

	return fully_conneted(f, ch_out, scope = "fc2")

def resnet50classifier(x, is_training, ch_in = 3, ch_out = 2, ch = 64, embedding_output = False, feature3_output = False):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False, activation = "linear")
	x1 = x 
	features = resnet_template(x, is_training = is_training, res_n = 50, ch = ch)
	features = [features[i] for i in range(len(features))]
	enc_dim = [ch*4, ch*2*4, ch*4*4, ch*8*4]
	# 4x 8x 16x 32x 

	g,_,_ = common.create_conv_layer('mid_1', features[3], enc_dim[3], enc_dim[3]//2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	g,_,_ = common.create_conv_layer('mid_2', g, enc_dim[3]//2, enc_dim[3]//4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True) 
	g = tf.reduce_mean(g, axis = [1,2])

	f = g
	f = fully_conneted(f, ch * 8, scope = "fc1")
	f = tf.nn.relu(f)
	
	if embedding_output:
		if feature3_output:
			return f, features[3]
		else:
			return f

	return fully_conneted(f, ch_out, scope = "fc2")

def FCNclassifier(x, is_training, ch_in = 3, ch_out = 2, ch = 64, embedding_output = False):
	x, _, _ = common.create_conv_layer('enc_1', x, ch_in, ch, kx = 7, ky = 7, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = False)

	x, _, _ = common.create_conv_layer('enc_2', x, ch, ch * 2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_3', x, ch * 2, ch * 2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	x, _, _ = common.create_conv_layer('enc_4', x, ch*2, ch * 4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_5', x, ch*4, ch * 4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	x, _, _ = common.create_conv_layer('enc_6', x, ch*4, ch * 8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_7', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	x, _, _ = common.create_conv_layer('enc_8', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_9', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	x, _, _ = common.create_conv_layer('enc_10', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_11', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	x, _, _ = common.create_conv_layer('enc_12', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = is_training, batchnorm = True)
	x, _, _ = common.create_conv_layer('enc_13', x, ch*8, ch * 8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = is_training, batchnorm = True)
	
	g = tf.reduce_mean(x, axis = [1,2])

	f = g
	f = fully_conneted(f, ch * 8, scope = "fc1")
	f = tf.nn.relu(f)
	
	if embedding_output:
		return f 

	return fully_conneted(f, ch_out, scope = "fc2")


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

