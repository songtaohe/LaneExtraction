
import numpy as np 
import threading
import scipy.ndimage 
from time import time 
import random 
import cv2 
import json
import math

global_lock = threading.Lock()

def rotate(pos, angle, size):
	x = pos[0] - int(size/2)
	y = pos[1] - int(size/2)

	new_x = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
	new_y = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))

	return (int(new_x + int(size/2)), int(new_y + int(size/2)))



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
		self.links = []
		self.maxbatchsize = 128
		self.image_batch = np.zeros((self.maxbatchsize, image_size, image_size,3))
		self.normal_batch = np.zeros((self.maxbatchsize, image_size, image_size,2))
		self.target_batch = np.zeros((self.maxbatchsize, image_size, image_size,1))
		self.target_t_batch = np.zeros((self.maxbatchsize, image_size, image_size,1))
		self.connector_batch = np.zeros((self.maxbatchsize, image_size, image_size,7))
		self.target_label_batch = np.zeros((self.maxbatchsize, 1))
		self.mask_batch = np.zeros((self.maxbatchsize, image_size, image_size,1))
		
		self.testing = testing

		self.poscode = np.zeros((image_size * 2, image_size * 2,2))
		for i in range(image_size * 2):
			self.poscode[i,:,0] = float(i) / image_size - 1.0
			self.poscode[:,i,1] = float(i) / image_size - 1.0
			


	def preload(self, ind = None):
		# global global_lock

		# global_lock.acquire()
		# for laneMap in self.laneMaps:
		# 	laneMap.save(self.fgt_folder)
		# self.laneMaps = []
		# global_lock.release()
		self.links = []
		for i in range(self.preload_tiles if ind is None else 1):
			while True:
				ind = random.choice(self.indrange)# if ind is None else ind 
				links = json.load(open(self.folder+"/link%s.json" % ind))

				if len(links[2]) == 0:
					continue

				


				sat_img = scipy.ndimage.imread(self.folder+"/sat%s.jpg" % ind)
				mask = scipy.ndimage.imread(self.folder+"/regionmask%s.jpg" % ind)
				target = scipy.ndimage.imread(self.folder+"/lane%s.jpg" % ind)
				#target_t = scipy.ndimage.imread(self.folder+"/terminal%s.jpg" % ind)
				normal = scipy.ndimage.imread(self.folder+"/normal%s.jpg" % ind)
				
				
				#target_t = cv2.GaussianBlur(target_t, (5,5), 1.0)

				if len(np.shape(mask)) == 3:
					mask = mask[:,:,0]
				
				if len(np.shape(target)) == 3:
					target = target[:,:,0]

				#if len(np.shape(target_t)) == 3:
				#	target_t = target_t[:,:,0]

				angle = 0
				if self.testing == False and random.randint(0,5) < 4:
					angle = random.randint(0,3) * 90 + random.randint(-30,30)
					#angle = 10

					sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
					mask = scipy.ndimage.rotate(mask, angle, reshape=False)
					target = scipy.ndimage.rotate(target, angle, reshape=False)
					#target_t = scipy.ndimage.rotate(target_t, angle, reshape=False)
					normal = scipy.ndimage.rotate(normal, angle, reshape=False, cval=127)

					# rotate links
					nidmap, nodes, locallinks = links
					# locallinks.append([newvertices, st, ed, st_nid, ed_nid])
					#locallinks = links
					newlocallinks = []
					for locallink in locallinks:
						oor = False
						newlocallink = []
						for k in range(len(locallink)):
							pos = [locallink[k][0], locallink[k][1]]
							pos = rotate(pos, -angle, self.datasetImageSize) 
							if pos[0] < 0 or pos[0] > self.datasetImageSize or pos[1] < 0 or pos[1] > self.datasetImageSize:
								oor = True
								break
							
							newlocallink.append(pos)

						if oor == False:
							newlocallinks.append(newlocallink)
					
					if len(newlocallinks) == 0:
						continue 
					
					links[2] = newlocallinks
							
					new_nodes = {}
					for k in nodes.keys():
						pos = nodes[k]
						pos = rotate(pos, -angle, self.datasetImageSize)
						if pos[0] < 0 or pos[0] > self.datasetImageSize or pos[1] < 0 or pos[1] > self.datasetImageSize:
							continue
						new_nodes[k] = pos

					if len(new_nodes) == 0:
						continue 

					links[1] = new_nodes

				

				normal = (normal.astype(np.float) - 127) / 127.0
				normal = normal[:,:,1:3] # cv2 is BGR scipy and Image PIL are RGB

				normal_x = normal[:,:,1]
				normal_y = normal[:,:,0]

				new_normal_x = normal_x * math.cos(math.radians(-angle)) - normal_y * math.sin(math.radians(-angle))
				new_normal_y = normal_x * math.sin(math.radians(-angle)) + normal_y * math.cos(math.radians(-angle))

				normal[:,:,0] = new_normal_x
				normal[:,:,1] = new_normal_y


				
				sat_img = sat_img.astype(np.float) / 255.0 - 0.5 
				mask = mask.astype(np.float) / 255.0 
				target = target.astype(np.float) / 255.0
				#target_t = target_t.astype(np.float) / 255.0
				
				self.links.append(links)

				self.images[i,:,:,:] = sat_img
				self.masks[i,:,:,0] = mask
				self.targets[i,:,:,0] = target
				#self.targets_t[i,:,:,0] = target_t
				self.normal[i,:,:,:] = normal

				# augmentation on images 
				if self.testing == False:
					self.images[i,:,:,:] = self.images[i,:,:,:] * (0.8 + 0.2 * random.random()) - (random.random() * 0.4 - 0.2)
					self.images[i,:,:,:] = np.clip(self.images[i,:,:,:], -0.5, 0.5)

					self.images[i,:,:,0] = self.images[i,:,:,0] * (0.8 + 0.2 * random.random())
					self.images[i,:,:,1] = self.images[i,:,:,1] * (0.8 + 0.2 * random.random())
					self.images[i,:,:,2] = self.images[i,:,:,2] * (0.8 + 0.2 * random.random())			

				break
		
		self.getBatchInternal(self.maxbatchsize)

	def getBatchInternal(self, batchsize):
		#print("getting batch")

		img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		connector1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		connector2 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		connectorlink = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		
		for i in range(batchsize):
			while True:
				tile_id = random.randint(0,self.preload_tiles-1)
				nidmap, nodes, locallinks = self.links[tile_id]
				if len(locallinks) == 0:
					continue

				# sample two in-connected points
				# sample two connected points
				coin = random.randint(0,1)

				# force coin to be only 0 
				coin = 0

				
				#print(i, tile_id, coin)
				if coin == 0:
					locallink = random.choice(locallinks)
					vertices = locallink

					sr = (vertices[0][1] + vertices[-1][1]) // 2
					sc = (vertices[0][0] + vertices[-1][0]) // 2

					# sr += random.randint(-50,50)
					# sc += random.randint(-50,50)

					sr -= self.image_size // 2
					sc -= self.image_size // 2

					if sr < 0: 
						sr = 0 
					if sr + self.image_size >= self.datasetImageSize:
						sr = self.datasetImageSize - self.image_size

					if sc < 0: 
						sc = 0 
					if sc + self.image_size >= self.datasetImageSize:
						sc = self.datasetImageSize - self.image_size
					
					img = img * 0
					#connector = connector * 0

					connector1 = connector1 * 0
					connector2 = connector2 * 0


					#st = random.randint(st-1,st+1)
					#ed = random.randint(ed-1,ed+1)

					# if st < 0:
					# 	st = 0

					# if ed >= len(vertices):
					# 	ed = len(vertices) - 1

					for k in range(len(vertices)-1):
						x1 = vertices[k][0] - sc 
						y1 = vertices[k][1] - sr 
						x2 = vertices[k+1][0] - sc 
						y2 = vertices[k+1][1] - sr 

						cv2.line(img, (x1,y1), (x2,y2), (255), 5)

						if k == 0:
							cv2.circle(connector1, (x1,y1), 12, (255), -1)
							xx1, yy1 = x1, y1
							#print(x1,y1)
						if k == len(vertices)-2:
							xx2, yy2 = x2, y2
							cv2.circle(connector2, (x2,y2), 12, (255), -1)
							#print(x2,y2)

					x1,y1 = xx1,yy1
					x2,y2 = xx2,yy2

					if x1 < 0 or x1 >= self.image_size or x2 < 0 or x2 >= self.image_size:
						continue
					if y1 < 0 or y1 >= self.image_size or y2 < 0 or y2 >= self.image_size:
						continue
					
					#connectorlink *= 0
					#cv2.line(connectorlink, (x1,y1), (x2,y2), (255),8)


					self.target_batch[i,:,:,0] = np.copy(img) / 255.0

					self.connector_batch[i,:,:,0] = np.copy(connector1) / 255.0 - 0.5
					self.connector_batch[i,:,:,3] = np.copy(connector2) / 255.0 - 0.5
					self.connector_batch[i,:,:,1:3] = self.poscode[self.image_size - y1:self.image_size*2 - y1, self.image_size - x1:self.image_size*2 - x1, :]
					self.connector_batch[i,:,:,4:6] = self.poscode[self.image_size - y2:self.image_size*2 - y2, self.image_size - x2:self.image_size*2 - x2, :]
					self.connector_batch[i,:,:,6] = np.copy(connectorlink) / 255.0 - 0.5

					self.target_label_batch[i,0] = 1
					self.image_batch[i,:,:,:] = self.images[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
					#self.target_t_batch[i,:,:,0] = self.targets_t[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, 0] 
					self.normal_batch[i,:,:,:] = self.normal[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
					
				else:
					nid1 = random.choice(nodes.keys())
					candidate = []
					pos1 = nodes[nid1]
					for nid2, pos2 in nodes.items():
						if nid2 == nid1:
							continue

						if nid2 in nidmap[nid1]:
							continue
						
						r = 8 * 40
						D = (pos2[0] - pos1[0]) ** 2 + abs(pos2[1] - pos1[1]) ** 2
						if D > r**2:
							continue
						
						candidate.append([nid2, pos2])
					
					if len(candidate) == 0:
						continue

					nid2, pos2 = random.choice(candidate)

					sr = (pos1[1] + pos2[1]) // 2
					sc = (pos1[0] + pos2[0]) // 2

					# sr += random.randint(-50,50)
					# sc += random.randint(-50,50)

					sr -= self.image_size // 2
					sc -= self.image_size // 2

					if sr < 0: 
						sr = 0 
					if sr + self.image_size >= self.datasetImageSize:
						sr = self.datasetImageSize - self.image_size

					if sc < 0: 
						sc = 0 
					if sc + self.image_size >= self.datasetImageSize:
						sc = self.datasetImageSize - self.image_size
							

					img = img * 0
					connector1 = connector1 * 0
					connector2 = connector2 * 0

					x1 = pos1[0] - sc 
					y1 = pos1[1] - sr 
					x2 = pos2[0] - sc 
					y2 = pos2[1] - sr

					#print(x1,y1,x2,y2)

					cv2.circle(connector1, (x1,y1), 12, (255), -1)
					cv2.circle(connector2, (x2,y2), 12, (255), -1)

					#connectorlink *= 0
					#cv2.line(connectorlink, (x1,y1), (x2,y2), (255),8)


					self.connector_batch[i,:,:,0] = np.copy(connector1) / 255.0 - 0.5
					self.connector_batch[i,:,:,3] = np.copy(connector2) / 255.0 - 0.5
					self.connector_batch[i,:,:,1:3] = self.poscode[self.image_size - y1:self.image_size*2 - y1, self.image_size - x1:self.image_size*2 - x1, :]
					self.connector_batch[i,:,:,4:6] = self.poscode[self.image_size - y2:self.image_size*2 - y2, self.image_size - x2:self.image_size*2 - x2, :]
					self.connector_batch[i,:,:,6] = np.copy(connectorlink) / 255.0 - 0.5



					self.target_batch[i,:,:,0] = np.copy(img) / 255.0
					#self.connector_batch[i,:,:,0] = np.copy(connector) / 255.0
					self.target_label_batch[i,0] = 0

					self.image_batch[i,:,:,:] = self.images[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
					#self.target_t_batch[i,:,:,0] = self.targets_t[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, 0] 
					self.normal_batch[i,:,:,:] = self.normal[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
					
				#print("we reach here")		
				break
		
		#print("getting batch done")
		return self.image_batch[:batchsize, :,:,:], self.connector_batch[:batchsize,:,:,:], self.target_batch[:batchsize, :,:,:], self.target_label_batch[:batchsize,:], self.normal_batch[:batchsize,:,:,:]

	def getBatch(self, batchsize):
		st = random.randint(0, self.maxbatchsize - batchsize - 1)

		return self.image_batch[st:st+batchsize, :,:,:], self.connector_batch[st:st+batchsize,:,:,:], self.target_batch[st:st+batchsize, :,:,:], self.target_label_batch[st:st+batchsize,:], self.normal_batch[st:st+batchsize,:,:,:]



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


