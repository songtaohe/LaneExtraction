from dataloader import  ParallelDataLoader
from model import LaneModel

import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from framework.training import TrainingFramework

from PIL import Image 
import numpy as np 
from subprocess import Popen 
import math
import json 

class Train(TrainingFramework):
	def __init__(self):
		self.image_size = 640
		self.batch_size = 4
		self.datafolder = "../dataset_training"
		self.training_range = []
		self.use_sdmap = False
		self.backbone = sys.argv[1] # resnet34v3

		dataset_split = json.load(open("../split_all.json"))
		
		for tid in dataset_split["training"]:
			for i in range(9):
				self.training_range.append("_%d" % (tid*9+i))
	
		self.instance = "_laneExtraction_run1_640_%s_500ep" % self.backbone 
		if self.use_sdmap:
			self.instance += "_withsdmap"

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
		self.model = LaneModel(sess, self.image_size, batchsize=self.batch_size, sdmap = self.use_sdmap, backbone = self.backbone)
		return self.model

	def getBatch(self, dataloader):
		return dataloader.getBatch(self.batch_size)

	def train(self, batch, lr):
		self.counter += 1 
		ret = self.model.train(batch[0], batch[1], batch[2],batch[3], lr, sdmap = batch[-1])
	
		return ret 


	def preload(self, dataloader, step):
		if step > 0 and step % 50 == 0:
			dataloader.preload()


	# placeholder methods
	def getLoss(self, result):
		if math.isnan(result[0]):
			print("loss is nan ...")
			exit()

		return result[0]

	def getProgress(self, step):
		return step / float(self.epochsize)

	def saveModel(self, step):
		if step % (self.epochsize * 10) == 0:
			self.model.saveModel(self.modelfolder + "/model%d" % (step // (self.epochsize)))
		return False

	def visualization(self, step, result = None, batch = None):
		direction_img = np.zeros((self.image_size, self.image_size, 3))
		if step % 100 == 0:
			ind = ((step // 100) * self.batch_size) % 128
			#batch[3] = np.clip(batch[3], -1, 1)
			#result[1] = np.clip(result[1], -1, 1)

			for i in range(self.batch_size):
				Image.fromarray(((batch[0][i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save(self.validationfolder + "/input%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,0]) * 255).astype(np.uint8) ).save(self.validationfolder + "/mask%d.jpg" % (ind+i))
				Image.fromarray(((batch[2][i,:,:,0]) * 255).astype(np.uint8) ).save(self.validationfolder + "/target%d.jpg" % (ind+i))
				
				Image.fromarray(((batch[4][i,:,:,0]) * 255).astype(np.uint8) ).save(self.validationfolder + "/sdmap%d.jpg" % (ind+i))
				

				direction_img[:,:,2] = batch[3][i,:,:,0] * 127 + 127
				direction_img[:,:,1] = batch[3][i,:,:,1] * 127 + 127
				direction_img[:,:,0] = 127

				Image.fromarray(direction_img.astype(np.uint8)).save(self.validationfolder + "/targe_direction%d.jpg" % (ind+i))
				
				Image.fromarray(((result[1][i,:,:,0]) * 255).astype(np.uint8) ).save(self.validationfolder + "/output%d.jpg" % (ind+i))
				
				direction_img[:,:,2] = np.clip(result[1][i,:,:,1],-1,1) * 127 + 127
				direction_img[:,:,1] = np.clip(result[1][i,:,:,2],-1,1) * 127 + 127
				direction_img[:,:,0] = 127
				
				Image.fromarray(direction_img.astype(np.uint8) ).save(self.validationfolder + "/output_direction%d.jpg" % (ind+i))
				
			
		return False


if __name__ == "__main__":
	trainer = Train()
	epochsisze = trainer.epochsize

	config = {}
	config["learningrate"] = 0.001
	config["lr_decay"] = [0.1,0.1]
	config["lr_decay_step"] = [epochsisze * 350,epochsisze * 450]
	config["step_init"] = 0
	config["step_max"] = epochsisze * 500 + 1
	config["use_validation"] = False
	config["logfile"] = "log_lane_%s.json" % trainer.instance

	trainer.run(config)
