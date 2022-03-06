import os 
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import tensorflow as tf 
import sys 
from cnnmodels.resnet34unet import resnet34unet_v3, unet, unet_dilated

class LaneModel():
	def __init__(self, sess, size = 640, batchsize = 4, sdmap = False, backbone = "resnet34v3"):
		self.sess = sess 
		self.batchsize = batchsize
		self.input = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 3])
		self.target = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 1])
		self.target_normal = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 2])
		self.sdmap = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 1])
		self.backbone = backbone
		self.mask = tf.placeholder(dtype = tf.float32, shape = [None, size, size, 1])
		
		
		self.lr = tf.placeholder(dtype=tf.float32)
		self.is_training = tf.placeholder(tf.bool, name="istraining")

		self.hassdmap = sdmap

		# if sdmap:
		# 	output = resnet34unet_v3(tf.concat([self.input, self.sdmap],axis=3), self.is_training, ch_in = 4, ch_out = 4)
		# else:
		
		if self.backbone == "resnet34v3":
			output = resnet34unet_v3(self.input, self.is_training, ch_in = 3, ch_out = 4)
		elif self.backbone == "resnet18v3":
			output = resnet34unet_v3(self.input, self.is_training, ch_in = 3, ch_out = 4, res_n = 18)
		elif self.backbone == "unet":
			output = unet(self.input, self.is_training, ch_in = 3, ch_out = 4)
		elif self.backbone == "unet_dilated":
			output = unet_dilated(self.input, self.is_training, ch_in = 3, ch_out = 4)

		self.loss = self.singlescaleloss(output[:,:,:,0:2], self.target, self.mask)
		self.loss += tf.reduce_mean(self.mask * tf.square(self.target_normal - output[:,:,:,2:4]))

		#self.loss += self.celoss(output[:,:,:,2:4], self.target_t, self.mask)
		
		output_seg = tf.nn.softmax(output[:,:,:,0:2])[:,:,:,0:1]
		output_direction = output[:,:,:,2:4]

		self.output = tf.concat([output_seg, output_direction], axis=3)

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		
		
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=21)
	
	def singlescaleloss(self, p, target, mask):
		t1 = target
		
		def ce_loss(p, t):
			#t = tf.concat([t,1-t], axis=3)
			pp0 = p[:,:,:,0:1]
			pp1 = p[:,:,:,1:2]

			loss =  - (t * pp0 + (1-t) * pp1 - tf.log(tf.exp(pp0) + tf.exp(pp1)))
			loss = tf.reduce_mean(loss * mask)
			return loss

		def dice_loss(p, t):
			#return 0
			p = tf.math.sigmoid(p[:,:,:,0:1] - p[:,:,:,1:2])
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

	def train(self, x_in, x_mask, target, target_normal, lr, sdmap = None):
		feed_dict = {
			self.input : x_in,
			self.mask : x_mask,
			self.target : target,
			self.target_normal : target_normal,
			self.lr : lr,
			self.is_training : True
		}
		if self.hassdmap:
			feed_dict[self.sdmap] = sdmap 

		ops = [self.loss, self.output, self.train_op]
		
		return self.sess.run(ops, feed_dict=feed_dict)

	def infer(self, x_in, sdmap = None):
		feed_dict = {
			self.input : x_in,
			self.is_training : False
		}

		if self.hassdmap:
			feed_dict[self.sdmap] = sdmap 

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

