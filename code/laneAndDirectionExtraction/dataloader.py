import numpy as np 
import threading
import scipy.ndimage 
from time import time 
import random 
import math

global_lock = threading.Lock()

class Dataloader():
	def __init__(self, folder, indrange, image_size = 640, datasetImageSize = 2048, preload_tiles = 4, testing = False):
		self.folder = folder
		self.indrange = indrange
		self.image_size = image_size 
		self.datasetImageSize = datasetImageSize
		self.preload_tiles = preload_tiles
		self.images = np.zeros((preload_tiles, datasetImageSize, datasetImageSize,3))
		self.normal = np.zeros((preload_tiles, datasetImageSize, datasetImageSize,2))
		self.targets = np.zeros((preload_tiles, datasetImageSize, datasetImageSize,1))
		self.targets_t = np.zeros((preload_tiles, datasetImageSize, datasetImageSize,1))
		self.masks = np.ones((preload_tiles, datasetImageSize, datasetImageSize,1))
		self.sdmaps = np.ones((preload_tiles, datasetImageSize, datasetImageSize,1))
		

		self.image_batch = np.zeros((8, image_size, image_size,3))
		self.normal_batch = np.zeros((8, image_size, image_size,2))
		self.target_batch = np.zeros((8, image_size, image_size,1))
		self.target_t_batch = np.zeros((8, image_size, image_size,1))
		self.sdmap_batch = np.zeros((8, image_size, image_size,1))
		
		self.mask_batch = np.zeros((8, image_size, image_size,1))
		
		self.testing = testing

	def preload(self, ind = None):
		# global global_lock

		# global_lock.acquire()
		# for laneMap in self.laneMaps:
		# 	laneMap.save(self.fgt_folder)
		# self.laneMaps = []
		# global_lock.release()

		for i in range(self.preload_tiles if ind is None else 1):
			ind = random.choice(self.indrange) if ind is None else ind 
			sat_img = scipy.ndimage.imread(self.folder+"/sat%s.jpg" % ind)
			mask = scipy.ndimage.imread(self.folder+"/regionmask%s.jpg" % ind)
			target = scipy.ndimage.imread(self.folder+"/lane%s.jpg" % ind)
			#target_t = scipy.ndimage.imread(self.folder+"/terminal%s.jpg" % ind)
			normal = scipy.ndimage.imread(self.folder+"/normal%s.jpg" % ind)
			#sdmap = scipy.ndimage.imread(self.folder+"/sdmap%s.jpg" % ind)

			#target_t = cv2.GaussianBlur(target_t, (5,5), 1.0)

			if len(np.shape(mask)) == 3:
				mask = mask[:,:,0]
			
			if len(np.shape(target)) == 3:
				target = target[:,:,0]

			# if len(np.shape(sdmap)) == 3:
			# 	sdmap = sdmap[:,:,0]

			# if len(np.shape(target_t)) == 3:
			# 	target_t = target_t[:,:,0]

			angle = 0
			if self.testing == False and random.randint(0,5) < 4:
				angle = random.randint(0,3) * 90 + random.randint(-30,30)
				#angle = 10

				sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
				mask = scipy.ndimage.rotate(mask, angle, reshape=False)
				target = scipy.ndimage.rotate(target, angle, reshape=False)
				#sdmap = scipy.ndimage.rotate(sdmap, angle, reshape=False)
				# target_t = scipy.ndimage.rotate(target_t, angle, reshape=False)
				normal = scipy.ndimage.rotate(normal, angle, reshape=False, cval=127)


			normal = (normal.astype(np.float) - 127) / 127.0
			normal = normal[:,:,1:3] # cv2 is BGR scipy and Image PIL are RGB

			normal_x = normal[:,:,1]
			normal_y = normal[:,:,0]

			new_normal_x = normal_x * math.cos(math.radians(-angle)) - normal_y * math.sin(math.radians(-angle))
			new_normal_y = normal_x * math.sin(math.radians(-angle)) + normal_y * math.cos(math.radians(-angle))

			normal[:,:,0] = new_normal_x
			normal[:,:,1] = new_normal_y
			normal = np.clip(normal, -0.9999, 0.9999)
				
			
			sat_img = sat_img.astype(np.float) / 255.0 - 0.5 
			mask = mask.astype(np.float) / 255.0 
			target = target.astype(np.float) / 255.0
			#sdmap = sdmap.astype(np.float) / 255.0
			# target_t = target_t.astype(np.float) / 255.0
			

			self.images[i,:,:,:] = sat_img
			self.masks[i,:,:,0] = mask
			self.targets[i,:,:,0] = target
			# self.targets_t[i,:,:,0] = target_t
			self.normal[i,:,:,:] = normal
			#self.sdmaps[i,:,:,0] = sdmap
			
			# augmentation on images 
			if self.testing == False:
				self.images[i,:,:,:] = self.images[i,:,:,:] * (0.8 + 0.2 * random.random()) - (random.random() * 0.4 - 0.2)
				self.images[i,:,:,:] = np.clip(self.images[i,:,:,:], -0.5, 0.5)

				self.images[i,:,:,0] = self.images[i,:,:,0] * (0.8 + 0.2 * random.random())
				self.images[i,:,:,1] = self.images[i,:,:,1] * (0.8 + 0.2 * random.random())
				self.images[i,:,:,2] = self.images[i,:,:,2] * (0.8 + 0.2 * random.random())			



	def getBatch(self, batchsize):
		for i in range(batchsize):
			while True:
				tile_id = random.randint(0,self.preload_tiles-1)
				x = random.randint(0, self.datasetImageSize-1-self.image_size)
				y = random.randint(0, self.datasetImageSize-1-self.image_size)

				if np.sum(self.targets[tile_id, x+64:x+self.image_size-64, y+64:y+self.image_size-64,:]) < 100:
					continue
				
				if np.sum(self.masks[tile_id, x+64:x+self.image_size-64, y+64:y+self.image_size-64,:]) < 50*50:
					continue
				
				self.image_batch[i,:,:,:] = self.images[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				self.mask_batch[i,:,:,:] = self.masks[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				self.target_batch[i,:,:,:] = self.targets[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				self.target_t_batch[i,:,:,:] = self.targets_t[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				self.normal_batch[i,:,:,:] = self.normal[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				#self.sdmap_batch[i,:,:,:] = self.sdmaps[tile_id, x:x+self.image_size, y:y+self.image_size,:]
				break
		

		return self.image_batch[:batchsize, :,:,:], self.mask_batch[:batchsize,:,:,:], self.target_batch[:batchsize, :,:,:], self.normal_batch[:batchsize,:,:,:], self.sdmap_batch[:batchsize,:,:,:]



class ParallelDataLoader():
	def __init__(self, *args,**kwargs):
		self.n = 4
		self.subloader = []
		self.subloaderReadyEvent = []
		self.subloaderWaitEvent = []
		
		self.current_loader_id = 0 


		for i in range(self.n):
			self.subloader.append(Dataloader(*args,**kwargs))
			self.subloaderReadyEvent.append(threading.Event())
			self.subloaderWaitEvent.append(threading.Event())

		for i in range(self.n):
			self.subloaderReadyEvent[i].clear()
			self.subloaderWaitEvent[i].clear()
		for i in range(self.n):
			x = threading.Thread(target=self.daemon, args=(i,))
			x.start() 


	def daemon(self, tid):
		c = 0

		while True:
			# 
			t0 = time()
			print("thread-%d starts preloading" % tid)
			self.subloader[tid].preload(None)
			
			self.subloaderReadyEvent[tid].set()

			print("thread-%d finished preloading (time = %.2f)" % (tid, time()-t0))

			self.subloaderWaitEvent[tid].wait()
			self.subloaderWaitEvent[tid].clear()

			if c == 0 and tid == 0:
				self.subloaderWaitEvent[tid].wait()
				self.subloaderWaitEvent[tid].clear()
			
			c = c + 1


	def preload(self):
		# release the current one 
		self.subloaderWaitEvent[self.current_loader_id].set()

		self.current_loader_id = (self.current_loader_id + 1) % self.n
		
		self.subloaderReadyEvent[self.current_loader_id].wait()
		self.subloaderReadyEvent[self.current_loader_id].clear()


	def getBatch(self,batch_size):
		return self.subloader[self.current_loader_id].getBatch(batch_size)
	def current(self):
		return self.subloader[self.current_loader_id]


