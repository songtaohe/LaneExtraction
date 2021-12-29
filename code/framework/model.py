import tensorflow as tf 

class ModelFramework():
	def __init__(self):
		self.inputs = {}
		self.ops = {}

		pass
	def addInput(self, name, tensor):
		self.inputs[name] = tensor
		return tensor

	def addCommonInputs(self):
		self.inputs["lr"] = tf.placeholder(dtype=tf.float32)
		self.inputs["is_training"] = tf.placeholder(tf.bool, name="istraining")
		return self.inputs["lr"], self.inputs["is_training"]

	def addTrainOp(self, loss):
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.inputs["lr"]).minimize(loss)
		self.addOp("train_op", self.train_op)
		
	def addOp(self, name, tensor):
		self.ops[name] = tensor

	def run(self, inputs, ops):
		feed_dict ={}
		for k,v in inputs.items():
			feed_dict[self.inputs[k]] = v 
		return self.sess.run([self.ops[op] for op in ops], feed_dict = feed_dict)

	def init(self, sess):
		self.sess = sess
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=5)

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

	def loss_ce_dice(self, p, target, mask):
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



	