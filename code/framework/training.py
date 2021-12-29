from time import time 
import tensorflow as tf 
import sys 
import json

class TrainingFramework():
	def __init__(self):
		pass

	def run(self, config):
		lr = config["learningrate"]
		lr_decay = config["lr_decay"]
		lr_decay_step = config["lr_decay_step"]
		step = config["step_init"]
		maxstep = config["step_max"]
		last_step = -1
		if "logfile" in config:
			logfile = config["logfile"]
		else:
			logfile = "log.json"

		use_validation = config["use_validation"]
		
		self.kv = {}
		
		logs = {}

		def addlog(k,v,s):
			if k in logs:
				logs[k][0].append(s)
				logs[k][1].append(v)
			else:
				logs[k] = [[s],[v]]

				


		gpu_options = tf.GPUOptions(allow_growth=True)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			model = self.createModel(sess)
			dataloader = self.createDataloader("training")
			# if use_validation:
			# 	dataloader_validation = self.createDataloader("testing")

			loss = 0
			t_load = 0
			t_preload = 0
			t_train = 0 
			t_other = 0
			t0 = time()

			lastprogress = -1

			while True:
				t_other += time() - t0
				t0 = time()
				batch = self.getBatch(dataloader)
				t1 = time()
				result = self.train(batch, lr)
				t2 = time()
				self.preload(dataloader, step)
				t3 = time()

				t_load += t1-t0 
				t_train += t2-t1
				t_preload += t3-t2

				loss += self.getLoss(result)

				if step % 10 == 0:
					progress = self.getProgress(step)
					p = int((progress - int(progress)) * 50)
					loss /= 10.0

					addlog("loss", loss, step)

					s = ""
					ks = sorted(self.kv.keys())
					for k in ks:
						v = self.kv[k]
						if v[1] > 0:
							s += " %s:%E " % (k, v[0] / float(v[1]))
							addlog(k, v[0] / float(v[1]), step)
							self.kv[k] = [0,0]

					sys.stdout.write("\rstep %d epoch:%.2f "% (step, progress) + ">" * p + "-" * (51-p) + " loss %f time %f %f %f %f " % (loss, t_preload, t_load, t_train, t_other - t_preload - t_load - t_train) + s )
					sys.stdout.flush()	

					p = int((progress - int(progress)) * 100000)
					if int(progress) != int(lastprogress):
						print("time per epoch", t_other)
						print("eta", t_other * (maxstep - step) / max(1, (step - last_step)) / 3600.0 )
						last_step = step
						t_load = 0
						t_preload = 0
						t_train = 0 
						t_other = 0

						

					loss = 0
					lastprogress = progress 

				self.saveModel(step)
				self.visualization(step, result, batch)
				
				for i in range(len(lr_decay_step)):
					if step == lr_decay_step[i]:
						lr = lr * lr_decay[i]

				if step == maxstep + 1:
					break
				
				if step % 1000 == 0 and step > 0:
					json.dump(logs, open(logfile, "w"), indent=2)

				step += 1

	# features
	def logvalue(self, k, v):
		if k in self.kv:
			self.kv[k][0] = self.kv[k][0] + v
			self.kv[k][1] = self.kv[k][1] + 1
		else:
			self.kv[k] = [v, 1]
		
			 



	# virtual methods
	def createDataloader(self, mode):
		print("createDataloader not implemented")
		exit()

	def createModel(self, sess):
		print("createModel not implemented")
		exit()

	def getBatch(self, dataloader):
		print("getBatch not implemented")
		exit()

	def train(self, batch, lr):
		print("train not implemented")
		exit()

	def preload(self, dataloader, step):
		print("preload not implemented")
		exit()


	# placeholder methods
	def getLoss(self, result):
		return 0.0

	def getProgress(self, step):
		return 0.0

	def saveModel(self, step):
		return False

	def visualization(self, step, result = None, batch = None):
		return False

	