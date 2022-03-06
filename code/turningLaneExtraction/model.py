import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np 
import tensorflow as tf 
import sys 
from cnnmodels.resnet34unet import resnet34unet_v3

class LinkModel():
	def __init__(self, sess, size = 640, batchsize = 4):
		self.sess = sess 
		self.batchsize = batchsize
		self.input = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 3])
		self.connector = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 7])
		self.target_t = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 1])
		self.context = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 2])
		self.target = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 1])
		self.target_label = tf.placeholder(dtype = tf.float32, shape = [None, 1])
		
		self.position_code = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 2])
		self.position_code_np = np.zeros((batchsize, size, size, 2))

		for i in range(size):
			self.position_code_np[:,i,:,0] = float(i) / size
			self.position_code_np[:,:,i,0] = float(i) / size




		self.lr = tf.placeholder(dtype=tf.float32)
		self.is_training = tf.placeholder(tf.bool, name="istraining")
		num = len(tf.global_variables())
		input_data = tf.concat([self.input, self.connector, self.context, self.position_code], axis=3)
			
		with tf.variable_scope("seg"):
			output_seg = resnet34unet_v3(input_data, self.is_training, ch_in = 14, ch_out = 2, feature_out=False)
		self.output = tf.nn.softmax(output_seg)
		
		self.loss = self.singlescaleloss(output_seg[:,:,:,0:2], self.target, 1.0)
		
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		
		self.sess.run(tf.global_variables_initializer())
		
		self.saver = tf.train.Saver(max_to_keep=5)
	
	def singlescaleloss(self, p, target, mask, keepbatchdim = False):
		t1 = target
		
		def ce_loss(p, t):
			#t = tf.concat([t,1-t], axis=3)
			pp0 = p[:,:,:,0:1]
			pp1 = p[:,:,:,1:2]

			loss =  - (t * pp0 + (1-t) * pp1 - tf.log(tf.exp(pp0) + tf.exp(pp1)))
			if keepbatchdim:
				loss = tf.reduce_mean(loss * mask, axis=[1,2,3])
			else:
				loss = tf.reduce_mean(loss * mask)
			return loss

		def dice_loss(p, t):
			#return 0
			p = tf.math.sigmoid(p[:,:,:,0:1] - p[:,:,:,1:2])
			if keepbatchdim:
				numerator = 2 * tf.reduce_sum(p * t * mask, axis=[1,2,3])
				denominator = tf.reduce_sum((p+t) * mask, axis=[1,2,3]) + 1.0
			else:
				numerator = 2 * tf.reduce_sum(p * t * mask)
				denominator = tf.reduce_sum((p+t) * mask ) + 1.0

			return 1 - numerator / denominator

		loss = 0
		loss += ce_loss(p, t1) + dice_loss(p, t1) * 0.333
		
		return loss 

	def celoss(self, p, target, mask):
		t1 = target
		def ce_loss(p, t):
			#t = tf.concat([t,1-t], axis=3)
			pp0 = p[:,:,:,0:1]
			pp1 = p[:,:,:,1:2]

			loss =  - (t * pp0 + (1-t) * pp1 - tf.log(tf.exp(pp0) + tf.exp(pp1)))
			loss = tf.reduce_mean(loss * mask)
			return loss
		return ce_loss(p, t1)


	def train(self, x_in, x_connector, target, target_label, context, lr):
		feed_dict = {
			self.input : x_in,
			self.connector : x_connector,
			self.target : target,
			self.target_label : target_label,
			self.context:context,
			self.lr : lr,
			self.is_training : True,
			self.position_code : self.position_code_np
		}

		ops = [self.loss, self.output, self.train_op]# + list(self.atts) 
		
		return self.sess.run(ops, feed_dict=feed_dict)
	
	
	def infer(self, x_in, x_connector, context):
		feed_dict = {
			self.input : x_in,
			self.connector : x_connector,
			self.context:context,
			self.position_code : self.position_code_np,
			self.is_training : False
		}

		ops = [self.output]
		
		return self.sess.run(ops, feed_dict=feed_dict)

	def evaluate(self, x_in, x_mask, target):
		feed_dict = {
			self.input : x_in,
			self.inputpatch : x_in_patch,
			self.is_training : False
		}

		ops = [self.output]
		return self.sess.run(ops, feed_dict=feed_dict)

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

