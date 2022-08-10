import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import sys 

from model_link import LinkModel

from PIL import Image 
import numpy as np 
from subprocess import Popen 
import tensorflow as tf 
import math
import scipy.ndimage
import pickle
import cv2
import json
from segtograph.segtographfunc import segtograph
from link_model_v4.infer import InferEngine
from link_model_v0.infer import InferEngine as InferEngineSeg
from time import time 

def seg2link(seg, p1, p2):
	p1 = (p1[1], p1[0])
	p2 = (p2[1], p2[0])
	#print(p1,p2)
	graph = segtograph((seg * 255).astype(np.uint8), 32, 0, 0, 32)
	#print("graph", graph)

	if len(graph) == 0:
		return False, None 
	
	def distance(pos1, pos2):
		a = pos1[0] - pos2[0]
		b = pos1[1] - pos2[1]

		return math.sqrt(a*a + b*b)

	p1dist = 100
	p2dist = 100
	p1nid = None  
	p2nid = None 

	for nid, nei in graph.items():
		if len(nei) == 1:
			d1 = distance(nid, p1)
			if d1 < p1dist:
				p1dist = d1
				p1nid = nid 
			
			d2 = distance(nid, p2)
			if d2 < p2dist:
				p2dist = d2
				p2nid = nid 

	if p1nid is None or p2nid is None:
		return False, None 
	if p1nid == p2nid:
		return False, None 

	#print(p1nid, p2nid)

	newgraph = {}
	for nid, nei in graph.items():
		newnid = nid
		# print(type(newnid), newnid)
		# print(type(p1nid), p1nid)
		if newnid == p1nid:
			newnid = p1
		elif newnid == p2nid:
			newnid = p2 
		newnei = []
		for nn in nei:
			new_nn = nn 
			if new_nn == p1nid:
				new_nn = p1
			elif new_nn == p2nid:
				new_nn = p2

			newnei.append(new_nn)
		newgraph[newnid] = newnei 

	# find a path 
	link = []
	cur = p1
	failed = False
	while True:
		#print(cur, p1, p2)
		link.append(cur)
		if cur == p2:
			break
		nei = newgraph[cur]
		if len(link) == 1:
			cur = nei[0]
			if len(nei) > 1:
				failed = True
				break
		else:
			if len(nei) != 2:
				failed = True
				break
			
			if nei[0] == link[-2]:
				cur = nei[1]
			else:
				cur = nei[0]

	if failed == False:
		return True, link 

	# need intepolation 

	link2 = []
	cur = p2
	
	while True:
		link2.append(cur)
		if cur == p1:
			break
		nei = newgraph[cur]
		if len(link2) == 1:
			cur = nei[0]
			if len(nei) > 1:
				break
		else:
			if len(nei) != 2:
				break
			
			if nei[0] == link2[-2]:
				cur = nei[1]
			else:
				cur = nei[0]
	
	for i in range(len(link2)):
		link.append(link2[len(link2) -1 - i])

	return True, link 	




inputsat = scipy.ndimage.imread(sys.argv[1]).astype(np.float) / 255.0
#inputseg = scipy.ndimage.imread(sys.argv[2]) / 255.0
inputdirection = (scipy.ndimage.imread(sys.argv[2]).astype(np.float) - 127) / 127.0
inputgraph = pickle.load(open(sys.argv[3]))
outputfolder = sys.argv[4]

outputfolder_details = sys.argv[4] + "/details/"


ways = json.load(open(outputfolder + "ways.json", "r"))


cnninput = 640

Popen("mkdir -p " + outputfolder, shell=True).wait()
Popen("mkdir -p " + outputfolder_details, shell=True).wait()


margin = cnninput // 2

inputsat = (np.pad(inputsat, ((margin, margin),(margin, margin),(0,0)), 'constant') - 0.5) * 0.81
dim = np.shape(inputsat)

#inputseg = np.pad(inputseg, ((margin, margin),(margin, margin)), 'constant')
inputdirection = np.pad(inputdirection, ((margin, margin),(margin, margin),(0,0)), 'constant')

context = np.zeros((dim[0], dim[1], 2))
#connector = np.zeros((1, dim[0], dim[1], 7))

#context = np.zeros_like(inputsat)
#context[:,:,0] = inputseg
context[:,:,0] = inputdirection[:,:,2]
context[:,:,1] = inputdirection[:,:,1]

# find all endpoints
endpoints = []
for nid, nei in inputgraph.items():
	if len(nei) == 1:
		endpoints.append(nid)
print("found %d endpoints" % len(endpoints))

starting_nodes = set()
ending_nodes = set()

for way in ways:
	starting_nodes.add(tuple(way[0]))
	ending_nodes.add(tuple(way[-1]))

	
r = 70*8
pairs = []

for i in range(len(endpoints)):
	if endpoints[i] in ending_nodes:
		for j in range(len(endpoints)):
			if endpoints[j] in starting_nodes:
				nid1 = endpoints[i]
				nid2 = endpoints[j]

				a = nid1[0] - nid2[0]
				b = nid1[1] - nid2[1]
				d = math.sqrt(a*a + b*b)
				if d <= r:
					pairs.append((nid1, nid2))

print("found %d pairs" % len(pairs))

batchsize = 8


	
model = InferEngine(batchsize=batchsize)
image_size = cnninput

image_batch = np.zeros((batchsize, image_size, image_size,3))
direction_batch = np.zeros((batchsize, image_size, image_size,2))
connector_batch = np.zeros((batchsize, image_size, image_size,7))
	
img = np.zeros((image_size, image_size), dtype=np.uint8)

poscode = np.zeros((image_size * 2, image_size * 2,2))
for i in range(image_size * 2):
	poscode[i,:,0] = float(i) / image_size - 1.0
	poscode[:,i,1] = float(i) / image_size - 1.0


positions = []
results = []
# check links connection
t0 = time()
counter = 0
for i in range(0,len(pairs), batchsize):
	p = int(float(i) / len(pairs) * 60)
	sys.stdout.write("\rprogress " + "|" * p + '.' * (60-p) + " %d/%d time %.2f good link %d" % (i, len(pairs), time() - t0, counter))
	sys.stdout.flush()	

	for j in range(batchsize):
		if i + j < len(pairs):
			nid1, nid2 = pairs[i+j]
			pos1 = [nid1[1] + margin, nid1[0] + margin]
			pos2 = [nid2[1] + margin, nid2[0] + margin]

			start = [(pos1[0] + pos2[0])//2 - image_size//2, (pos1[1] + pos2[1])//2 - image_size//2]

			image_batch[j,:,:,:] = inputsat[start[1]:start[1] + image_size, start[0]:start[0] + image_size, :]
			direction_batch[j,:,:,:] = context[start[1]:start[1] + image_size, start[0]:start[0] + image_size, :]
		
			img *= 0
			cv2.circle(img, (pos1[0] - start[0], pos1[1] - start[1]), 12, (255), -1)
			connector_batch[j,:,:,0] = np.copy(img) / 255.0 - 0.5

			img *= 0
			cv2.circle(img, (pos2[0] - start[0], pos2[1] - start[1]), 12, (255), -1)
			connector_batch[j,:,:,3] = np.copy(img) / 255.0 - 0.5

			x1 = pos1[0] - start[0]
			y1 = pos1[1] - start[1]

			x2 = pos2[0] - start[0]
			y2 = pos2[1] - start[1]

			positions.append([[x1,y1], [x2,y2]])

			connector_batch[j,:,:,1:3] = poscode[image_size - y1:image_size*2 - y1, image_size - x1:image_size*2 - x1, :]
			connector_batch[j,:,:,4:6] = poscode[image_size - y2:image_size*2 - y2, image_size - x2:image_size*2 - x2, :]
			connector_batch[j,:,:,6] = -0.5



	
	ret =  model.infer(sat = image_batch, connector= connector_batch, direction=direction_batch)
	for j in range(batchsize):
		if i + j < len(pairs):
			results.append(ret[0][j,0])
			if ret[0][j,0] > 0.5:
				counter += 1
			# 
			Image.fromarray(((image_batch[j,:,:,:] + 0.5) * 255).astype(np.uint8)).save(outputfolder_details + "/aerial%d.png" % (i+j))
			Image.fromarray(((connector_batch[j,:,:,0:3]) * 127 + 127).astype(np.uint8) ).save(outputfolder_details + "/connectorA%d.png" % (i+j))
			Image.fromarray(((connector_batch[j,:,:,3:6]) * 127 + 127).astype(np.uint8) ).save(outputfolder_details + "/connectorB%d.png" % (i+j))
			
			direction_img = np.zeros((640, 640, 3))

			direction_img[:,:,2] = np.clip(direction_batch[j,:,:,0],-1,1) * 127 + 127
			direction_img[:,:,1] = np.clip(direction_batch[j,:,:,1],-1,1) * 127 + 127
			direction_img[:,:,0] = 127

			# direction_img[:,:,0] += batch[1][i,:,:,0] * 255 + 127
			# direction_img[:,:,1] += batch[1][i,:,:,3] * 255 + 127
			# direction_img[:,:,2] += batch[1][i,:,:,6] * 255 + 127
			
			direction_img = np.clip(direction_img, 0, 255)
			Image.fromarray(direction_img.astype(np.uint8) ).save(outputfolder_details + "/direction%d.png" % (i+j))
			
			Image.fromarray((ret[1][j,:,:,0] * 255).astype(np.uint8)).save(outputfolder_details + "/outputA%d.png" % (i+j))
			Image.fromarray((ret[1][j,:,:,1] * 255).astype(np.uint8)).save(outputfolder_details + "/outputB%d.png" % (i+j))

			json.dump([pairs[i+j], float(ret[0][j,0])], open(outputfolder_details+"/ret%d.json" % (i+j),"w"))



	
print("Found %d links" % counter)

tf.reset_default_graph()
model.sess.close()	


model = InferEngineSeg(batchsize = 1)
links = []
failed = 0
for i in range(len(pairs)):
	if results[i] > 0.5:
		nid1, nid2 = pairs[i]
		pos1 = [nid1[1] + margin, nid1[0] + margin]
		pos2 = [nid2[1] + margin, nid2[0] + margin]

		start = [(pos1[0] + pos2[0])//2 - image_size//2, (pos1[1] + pos2[1])//2 - image_size//2]

		image_batch[0,:,:,:] = inputsat[start[1]:start[1] + image_size, start[0]:start[0] + image_size, :]
		direction_batch[0,:,:,:] = context[start[1]:start[1] + image_size, start[0]:start[0] + image_size, :]
	
		img *= 0
		cv2.circle(img, (pos1[0] - start[0], pos1[1] - start[1]), 12, (255), -1)
		connector_batch[0,:,:,0] = np.copy(img) / 255.0 - 0.5

		img *= 0
		cv2.circle(img, (pos2[0] - start[0], pos2[1] - start[1]), 12, (255), -1)
		connector_batch[0,:,:,3] = np.copy(img) / 255.0 - 0.5

		x1 = pos1[0] - start[0]
		y1 = pos1[1] - start[1]

		x2 = pos2[0] - start[0]
		y2 = pos2[1] - start[1]

		#positions.append([[x1,y1], [x2,y2]])

		connector_batch[0,:,:,1:3] = poscode[image_size - y1:image_size*2 - y1, image_size - x1:image_size*2 - x1, :]
		connector_batch[0,:,:,4:6] = poscode[image_size - y2:image_size*2 - y2, image_size - x2:image_size*2 - x2, :]
		connector_batch[0,:,:,6] = -0.5
	
		seg = model.infer(sat=image_batch[0:1,:,:,:], connector = connector_batch[0:1,:,:,:], direction=direction_batch[0:1,:,:,:])

		Image.fromarray((seg[0,:,:,0]*255).astype(np.uint8))
		ok, link = seg2link(seg[0,:,:,0], [x1,y1], [x2,y2])
		newlink = []
		if not ok:
			#link = [[y1,x1], [y2,x2]]
			Image.fromarray((seg[0,:,:,0]*255).astype(np.uint8)).save("infer_link_v4_debug/seg%d.png" % i)
			# bezier curve

			failed += 1

		if ok:
			
			for pos in link:
				newpos = (int(pos[0] + start[1] - margin), int(pos[1] + start[0] - margin))
				newlink.append(newpos)
			
			links.append(newlink)

		
		Image.fromarray((seg[0,:,:,0]*255).astype(np.uint8)).save(outputfolder_details+"/seg%d.png" % i)
		json.dump([ok, newlink], open(outputfolder_details+"/link%d.json" % (i),"w"))
			

print(len(links), failed)
json.dump(links, open(outputfolder + "/links.json", "w"), indent=2)

			




