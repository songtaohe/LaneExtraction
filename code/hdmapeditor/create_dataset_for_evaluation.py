# 25.812813, -80.202372

import json 
import math 
from subprocess import Popen 

import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from PIL import Image 
import numpy as np 
import scipy.misc
import json 
import cv2
import pickle
from roadstructure import LaneMap 



regions = json.load(open(sys.argv[1]))
inputfolder = sys.argv[2]
outputfolder = sys.argv[3]

counter = 0 
counter_out = 0
total_length = 0 
for region in regions:
	min_lat, min_lon = region["lat"], region["lon"]
	region_tag = region["tag"]
	regionsize = 4096
	stride = 4096
	tilesize = 4096
	res = 8

	blocks = [region["ilat"],region["ilon"]]
	folder = inputfolder

	for ilat in range(blocks[0]):
		for ilon in range(blocks[1]):
			subregion = [min_lat + ilat * stride / res / 111111.0, min_lon + ilon * stride / res / 111111.0 / math.cos(math.radians(min_lat))]
			subregion += [min_lat + (ilat * stride + tilesize) / res / 111111.0, min_lon + (ilon * stride + tilesize) / res / 111111.0 / math.cos(math.radians(min_lat))]

			img = cv2.imread(folder + "/sat_%d.jpg" % (counter))
			try:
				labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
			except:
				break
			roadlabel, masklabel = labels 

			adv = 0
			for nid, p1 in roadlabel.nodes.items():
				for nn in roadlabel.neighbors[nid]:
					p2 = roadlabel.nodes[nn]
					L = math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)
					total_length += L 
					adv += L 

			print(adv/8/1000.0)

			polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			sr,sc = 0, 0 
			sat = img[sr:sr+4096, sc:sc+4096,:]
			mask = np.zeros_like(sat) + 255
			normal = np.zeros_like(sat) + 127
			lane = np.zeros_like(sat)
			margin = 128

			# draw mask
			for polygon in polygons:
				polygon_list = []
				for i in range(len(polygon)-1):
					x1 = masklabel.nodes[polygon[i]][0] - sc - margin
					y1 = masklabel.nodes[polygon[i]][1] - sr - margin
					x2 = masklabel.nodes[polygon[i+1]][0] - sc - margin
					y2 = masklabel.nodes[polygon[i+1]][1] - sr - margin
					
					polygon_list.append([x1,y1])
				polygon_list.append([x2,y2])

				area = np.array(polygon_list)
				area = area.astype(np.int32)
				cv2.fillPoly(mask, [area], (0,0,0))

			# draw lane and direction
			for nid, nei in roadlabel.neighbors.items():
				x1 = roadlabel.nodes[nid][0] - sc - margin
				y1 = roadlabel.nodes[nid][1] - sr - margin
				
				for nn in nei:
					if roadlabel.edgeType[(nid,nn)] != "way":
						continue
					
					x2 = roadlabel.nodes[nn][0] - sc - margin
					y2 = roadlabel.nodes[nn][1] - sr - margin

					dx = x2 - x1
					dy = y2 - y1
					l = math.sqrt(float(dx*dx + dy*dy)) + 0.001
					dx /= l 
					dy /= l

					
					color = (127 + int(dx * 127), 127 + int(dy * 127), 127)
					
					cv2.line(lane, (x1,y1), (x2,y2), (255,255,255), 5)
					cv2.line(normal, (x1,y1), (x2,y2), color, 5)
			

			cv2.imwrite(outputfolder + "/sat_%d.jpg" % (counter_out), sat)
			cv2.imwrite(outputfolder + "/regionmask_%d.jpg" % (counter_out), mask)
			cv2.imwrite(outputfolder + "/lane_%d.jpg" % (counter_out), lane)
			cv2.imwrite(outputfolder + "/normal_%d.jpg" % (counter_out), normal)

			counter_out += 1
			print(counter_out)

			
			
			
			counter += 1
			
print(total_length, total_length / 8 / 1000.0)


