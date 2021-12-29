from model import LinkModel
import tensorflow as tf 


import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

class InferEngine():
    def __init__(self, modelpath="../../models/TODO", batchsize = 8):
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model = LinkModel(self.sess, 640, batchsize=batchsize)
        self.model.restoreModel(modelpath)

    def infer(self, sat=None, connector=None, direction = None):
        return self.model.infer(sat, connector, direction)


