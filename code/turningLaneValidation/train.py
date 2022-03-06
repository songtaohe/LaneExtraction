from dataloader import Dataloader, ParallelDataLoader
from model import LinkModel

import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from framework.training import TrainingFramework



from PIL import Image 
import numpy as np 
from subprocess import Popen 
import tensorflow as tf 
import math
import json




class Train(TrainingFramework):
	def __init__(self, mode = "seg"):
		self.mode = mode
		self.image_size = 640
		self.batch_size = 8
		self.datafolder = "../dataset_training"
		self.training_range = []
		dataset_split = json.load(open("../split_all.json"))
		
		for tid in dataset_split["training"]:
			for i in range(9):
				self.training_range.append("_%d" % (tid*9+i))
	
		self.instance = "_turningLaneValidation_run1_640_resnet34_500ep"+self.mode
		
		#self.instance = "link_run6_640_resnet34v3" # gt direction
		self.modelfolder = "model" + self.instance
		self.validationfolder = "validation" + self.instance
		
		Popen("mkdir " + self.modelfolder, shell=True).wait()
		Popen("mkdir " + self.validationfolder, shell=True).wait()
		
		self.counter = 0
		self.disloss = 0

		self.epochsize = len(self.training_range) * 2048 * 2048 / (self.batch_size * self.image_size * self.image_size)
		
		pass

	def createDataloader(self, mode):
		self.dataloader = ParallelDataLoader(self.datafolder, self.training_range, image_size = self.image_size)
		self.dataloader.preload()
		return self.dataloader

	def createModel(self, sess):
		self.model = LinkModel(sess, self.image_size, batchsize=self.batch_size)
		
		return self.model

	def getBatch(self, dataloader):
		return dataloader.getBatch(self.batch_size)

	def train(self, batch, lr):
		self.counter += 1 

		ret = self.model.train(batch[0], batch[1], batch[2],batch[3],batch[4], lr)

		return ret


	def preload(self, dataloader, step):
		if step > 0 and step % 50 == 0:
			dataloader.preload()


	# placeholder methods
	def getLoss(self, result):
		if math.isnan(result[0]):
			print("loss is nan ...")
			exit()

		self.logvalue("segloss", result[1])
		self.logvalue("classloss", result[2])
			

		return result[0]

	def getProgress(self, step):
		return step / float(self.epochsize)

	def saveModel(self, step):
		if step % (self.epochsize * 5) == 0:
			self.model.saveModel(self.modelfolder + "/model%d" % step)
		return False

	def visualization(self, step, result = None, batch = None):
		direction_img = np.zeros((self.image_size, self.image_size, 3))
		

		if step % 100 == 0:
			ind = ((step // 100) * self.batch_size) % 128
			for i in range(self.batch_size):
				Image.fromarray(((batch[0][i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save(self.validationfolder + "/input%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,0:3]) * 127 + 127).astype(np.uint8) ).save(self.validationfolder + "/connector1%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,3:6]) * 127 + 127).astype(np.uint8) ).save(self.validationfolder + "/connector2%d.jpg" % (ind+i))
				
				Image.fromarray(((batch[2][i,:,:,1]) * 255).astype(np.uint8) ).save(self.validationfolder + "/target1%d.jpg" % (ind+i))
				Image.fromarray(((result[3][i,:,:,0]) * 255).astype(np.uint8) ).save(self.validationfolder + "/output1%d.jpg" % (ind+i))
				Image.fromarray(((batch[2][i,:,:,2]) * 255).astype(np.uint8) ).save(self.validationfolder + "/target2%d.jpg" % (ind+i))
				Image.fromarray(((result[3][i,:,:,1]) * 255).astype(np.uint8) ).save(self.validationfolder + "/output2%d.jpg" % (ind+i))

				with open(self.validationfolder + "/label%d.txt" % (ind+i), "w") as fout:
					fout.write("%f %f \n" % (batch[3][i,0], result[4][i,0]))

				

				def norm(x):
					#return x 
					amin = np.amin(x)
					amin = 0
					amax = np.amax(x)

					x = (x - amin) / max(0.00001, (amax - amin))
					return x

					
				direction_img[:,:,2] = np.clip(batch[4][i,:,:,0],-1,1) * 127 + 127
				direction_img[:,:,1] = np.clip(batch[4][i,:,:,1],-1,1) * 127 + 127
				direction_img[:,:,0] = 127

				direction_img[:,:,0] += batch[1][i,:,:,0] * 255 + 127
				direction_img[:,:,1] += batch[1][i,:,:,3] * 255 + 127
				direction_img[:,:,2] += batch[1][i,:,:,6] * 255 + 127
				
				direction_img = np.clip(direction_img, 0, 255)

				Image.fromarray(direction_img.astype(np.uint8) ).save(self.validationfolder + "/direction%d.jpg" % (ind+i))
				


		return False


if __name__ == "__main__":
	#trainer = Train(sys.argv[1])
	trainer = Train()
	
	epochsisze = trainer.epochsize

	config = {}
	config["learningrate"] = 0.0001
	config["lr_decay"] = [0.1,0.1]
	config["lr_decay_step"] = [epochsisze * 350,epochsisze * 450]
	
	config["step_init"] = 0
	config["step_max"] = epochsisze * 500 + 1
	
	config["use_validation"] = False
	config["logfile"] = "log_%s.json" % trainer.instance

	trainer.run(config)
