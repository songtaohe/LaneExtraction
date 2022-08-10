import json 
import sys 
import pickle 

import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from PIL import Image 
import numpy as np 
import scipy.misc
import scipy.ndimage
import cv2 
from time import time 
import math

class DirectionEvaluator():
	def __init__(self, dataset_folder, tileid):
		self.ways = json.load(open(dataset_folder + "/way%s.json" % tileid))
		self.groups = json.load(open(dataset_folder + "/group%s.json" % tileid))
		self.dataset_folder = dataset_folder
		self.tileid = tileid

	def loadFromDirectionMap(self, filename, roadtype="all"):
		direction = scipy.ndimage.imread(filename)
		direction = (direction.astype(np.float) - 127) / 127.0
		direction = direction[:,:,1:3]

		correct_length = 0 
		correct_cc = 0 
		total_length = 0 
		total_cc = 0 

		wayid = 0 
		for way in self.ways:
			if roadtype == "oneway" and self.groups[wayid][1] == False:
				wayid += 1
				continue 
			
			if roadtype == "twoway" and self.groups[wayid][1] == True:
				wayid += 1
				continue 

			cdot = 0
			for i in range(len(way)-1):
				dx = way[i+1][0] - way[i][0]
				dy = way[i+1][1] - way[i][1]

				d = math.sqrt(float(dx**2 + dy**2))

				dx = dx / d 
				dy = dy / d 
		
				N = int(d / 2)

				for j in range(N):
					alpha = float(j+1) / (N + 1)
					x = int(way[i+1][0] * alpha + way[i][0] * (1 - alpha))
					y = int(way[i+1][1] * alpha + way[i][1] * (1 - alpha))

					if x < 0  or x >= 4096 or y < 0 or y >= 4096:
						continue


					ddx = direction[y,x,1]
					ddy = direction[y,x,0]

					cdot += ddx * dx + ddy * dy 

			if len(way) <= 2:
				wayid += 1
				continue 
			
			if cdot > 0:
				correct_cc += 1
				correct_length += len(way)

			total_cc += 1
			total_length += len(way)
			wayid += 1

		# acc = float(correct_cc) / total_cc
		# acc_w = float(correct_length) / total_length
		# print(filename, acc, acc_w)
		return correct_length, correct_cc, total_length, total_cc  

	def inferDirections(self, inferEngine):
		correct_length = 0 
		correct_cc = 0 
		total_length = 0 
		total_cc = 0 

		padding = 256
		sat_img = scipy.ndimage.imread(self.dataset_folder+"/sat%s.jpg" % self.tileid)
		sat_img = np.pad(sat_img, [[padding, padding], [padding, padding], [0,0]], 'constant')
		
		target = scipy.ndimage.imread(self.dataset_folder+"/lane%s.jpg" % self.tileid)[:,:,0:1]
		target = target.astype(np.float) / 255.0

		normal = scipy.ndimage.imread(self.dataset_folder+"/normal%s.jpg" % self.tileid)
		normal = (normal.astype(np.float) - 127) / 127.0
		normal = normal[:,:,1:3] # cv2 is BGR scipy and Image PIL are RGB
		normal_x = normal[:,:,1]
		normal_y = normal[:,:,0]
		normal[:,:,0] = normal_x
		normal[:,:,1] = normal_y
		normal = np.clip(normal, -0.9999, 0.9999)

		targets = np.pad(target,[[padding,padding], [padding,padding], [0,0]], 'constant')
		directions = np.pad(normal,[[padding,padding], [padding,padding], [0,0]], 'constant')
				
		
		
		image_batch = np.zeros((256, 224, 224, 6+2+1))
		image_batch_reverse = np.zeros((256, 224, 224, 6+2+1))
		
		
		image_size = 224
		images = (sat_img.astype(np.float) / 255.0 - 0.5) * 0.81
		lane_raster = np.zeros((image_size, image_size), dtype=np.uint8)

		way_cc = 0
		for way in self.ways:
			ptr = 0
			for i in range(len(way)):
				if way[i][1] < 0 or way[i][1] >= 4096 or way[i][0] < 0 or way[i][0] >= 4096:
					continue

				r = way[i][1] - image_size // 2 + padding
				c = way[i][0] - image_size // 2 + padding
				image_batch[ptr,:,:,0:3] = images[r:r+image_size, c:c+image_size,:]
				image_batch_reverse[255-ptr,:,:,0:3] = images[r:r+image_size, c:c+image_size,:]

				image_batch[ptr,:,:,7:9] = directions[r:r+image_size, c:c+image_size,:]
				image_batch_reverse[255-ptr,:,:,7:9] = directions[r:r+image_size, c:c+image_size,:]

				image_batch[ptr,:,:,6:7] = targets[r:r+image_size, c:c+image_size,:]
				image_batch_reverse[255-ptr,:,:,6:7] = targets[r:r+image_size, c:c+image_size,:]

				if i < len(way)-1:
					dx = way[i+1][0] - way[i][0]
					dy = way[i+1][1] - way[i][1]

					l = math.sqrt(dx*dx + dy*dy + 1)
					dx /= l 
					dy /= l 
					direction = (dx,dy)
				else:
					dx,dy = direction

				image_batch[ptr,:,:,3] = dx 
				image_batch[ptr,:,:,4] = dy 

				image_batch_reverse[255-ptr,:,:,3] = -dx 
				image_batch_reverse[255-ptr,:,:,4] = -dy 


				lane_raster *= 0 
				bx = way[i][0]
				by = way[i][1]

				for j in range(max(0, i-3), min(len(way)-1, i+4)):
					x1 = way[j][0] - bx + image_size // 2
					y1 = way[j][1] - by + image_size // 2
					x2 = way[j+1][0] - bx + image_size // 2
					y2 = way[j+1][1] - by + image_size // 2
					cv2.line(lane_raster, (x1,y1), (x2,y2), (255), 8)

				image_batch[ptr,:,:,5] = np.copy(lane_raster).astype(np.float) / 255.0
				image_batch_reverse[255 - ptr,:,:,5] = np.copy(lane_raster).astype(np.float) / 255.0

				ptr += 1

			direction_score = inferEngine.infer(image_batch[:ptr,:,:,:])[0,0]
			#direction_score += inferEngine.infer(image_batch_reverse[255-ptr+1:,:,:,:])[0,1]
			#direction_score /= 2.0 

			# debug
			if self.tileid == "xxx_5":
				images_gif = []
				#images_gif2 = []
				
				imgall = np.zeros((224, 224*2,3), dtype=np.uint8) + 255
				last = None
				for i in range(ptr):
					

					direction_score_local = inferEngine.infer(image_batch[:i+1,:,:,:])[0,0]
					direction_score_local += inferEngine.infer(image_batch_reverse[255-i:,:,:,:])[0,1]
					direction_score_local /= 2.0 

					img = ((image_batch[i,:,:,0:3] + 0.5) * 255).astype(np.uint8)
					
					
					cv2.putText(img, "%d/%d %.3f/%.3f" % (i,ptr, direction_score_local, direction_score), (5,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
					x1,y1 = image_size//2, image_size//2
					dx = image_batch[i,0,0,3]
					dy = image_batch[i,0,0,4]

					x2,y2 = int(x1 + dx * 16), int(y1 + dy * 16)

					cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 2)
					cv2.circle(img, (x1,y1), 3,(0,0,255),-1)

					img = img.astype(np.float)
					img[:,:,1] += image_batch[i,:,:,5] * 64
					img = np.clip(img, 0, 255).astype(np.uint8)

					imgall[:,:224,:] = img


					if i == 0:
						y = 12 + int((1 - direction_score_local) * 200)
						cv2.circle(imgall, (240,y), 3,(0,0,255),-1)
						last = y
					
					else:
						y = 12 + int((1 - direction_score_local) * 200)
						cv2.line(imgall, (240 + int((float(i-1)/ptr)*200),last), (240 + int((float(i)/ptr)*200),y), (0,0,0),2)

						cv2.circle(imgall, (240 + int((float(i)/ptr)*200),y), 3,(0,0,255),-1)
						cv2.circle(imgall, (240 + int((float(i-1)/ptr)*200),last), 3,(0,0,255),-1)

						last = y


					images_gif.append(Image.fromarray(imgall))

				images_gif.append(images_gif[-1])
				images_gif.append(images_gif[-1])
				images_gif.append(images_gif[-1])
				
				images_gif[0].save(fp= "debug/example%d.gif" % (way_cc), format='GIF', append_images=images_gif,  save_all=True, duration=1000, loop=0)
			

			if direction_score < 0.5:
				correct_cc += 1
				correct_length += len(way)
			
			total_cc += 1
			total_length += len(way)

			way_cc += 1
			
		acc = float(correct_cc) / total_cc
		acc_w = float(correct_length) / total_length
		print(self.tileid, acc, acc_w)
		return correct_length, correct_cc, total_length, total_cc  


if __name__ == "__main__":
	tag = sys.argv[1]
	# python evaluate.py unet ...
	##### predict directions from segmentation 
	print("ALL ROADS")
	correct_length, correct_cc, total_length, total_cc = 0, 0, 0, 0
	
	for i in [0, 5, 6, 11, 12, 17, 18, 22, 25, 28, 31]:
		evaluator = DirectionEvaluator("../dataset_evaluation", "_%d" % i)
		ret = evaluator.loadFromDirectionMap("../all_results_%s/%d/direction.png" % (tag,i))
		correct_length += ret[0]
		correct_cc += ret[1]
		total_length += ret[2]
		total_cc += ret[3]

	acc = float(correct_cc) / total_cc
	acc_w = float(correct_length) / total_length	
	print("overall accuracy", acc, acc_w, total_length)

	# one way road
	print("ONE-WAY ROADS")
	correct_length, correct_cc, total_length, total_cc = 0, 0, 0, 0

	for i in [0, 5, 6, 11, 12, 17, 18, 22, 25, 28, 31]:
		evaluator = DirectionEvaluator("../dataset_evaluation", "_%d" % i)
		ret = evaluator.loadFromDirectionMap("../all_results_%s/%d/direction.png" % (tag,i), roadtype="oneway")
		correct_length += ret[0]
		correct_cc += ret[1]
		total_length += ret[2]
		total_cc += ret[3]

	acc = float(correct_cc) / total_cc
	acc_w = float(correct_length) / total_length	
	print("overall accuracy", acc, acc_w, total_length)

	# two way road
	print("TWO-WAY ROADS")
	correct_length, correct_cc, total_length, total_cc = 0, 0, 0, 0

	for i in [0, 5, 6, 11, 12, 17, 18, 22, 25, 28, 31]:
		evaluator = DirectionEvaluator("../dataset_evaluation", "_%d" % i)
		ret = evaluator.loadFromDirectionMap("../all_results_%s/%d/direction.png" % (tag,i), roadtype="twoway")
		correct_length += ret[0]
		correct_cc += ret[1]
		total_length += ret[2]
		total_cc += ret[3]

	acc = float(correct_cc) / total_cc
	acc_w = float(correct_length) / total_length	
	print("overall accuracy", acc, acc_w, total_length)

	


