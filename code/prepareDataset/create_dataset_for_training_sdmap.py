# 25.812813, -80.202372

import json 
import math 
from subprocess import Popen 

import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from satellite import mapbox as md
from osm import osm 
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

			# try:
			# 	labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
			# except:
			# 	break
			# roadlabel, masklabel = labels 

			OSMMap = osm.OSMLoader(subregion,noUnderground=True, includeServiceRoad=False, useblacklist = False)
			# TODO draw it 
			
			print(len(OSMMap.nodedict))
			
			#polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			for sr in [0, 1024, 2048]:
				for sc in [0, 1024, 2048]:
					sdmap = np.zeros((2048,2048), dtype=np.uint8)

					for node_id, node_info in OSMMap.nodedict.iteritems():
						lat1 = node_info["lat"]
						lon1 = node_info["lon"]
						#print(node_id, node_info["to"].keys() + node_info["from"].keys())
						for nid in node_info["to"].keys() + node_info["from"].keys():
							lat2 = OSMMap.nodedict[nid]["lat"]
							lon2 = OSMMap.nodedict[nid]["lon"]

							x1 = int((lon1 - subregion[1]) / (subregion[3] - subregion[1]) * 4096) - sc
							y1 = int((subregion[2] - lat1) / (subregion[2] - subregion[0]) * 4096) - sr
							
							x2 = int((lon2 - subregion[1]) / (subregion[3] - subregion[1]) * 4096) - sc
							y2 = int((subregion[2] - lat2) / (subregion[2] - subregion[0]) * 4096) - sr 

							#print(x1,y1,x2,y2)

							if (x1 >=0 and x1 <= 2048 and y1 >= 0 and y1 <= 2048) or (x2 >=0 and x2 <= 2048 and y2 >= 0 and y2 <= 2048) :
								cv2.line(sdmap, (x1,y1), (x2,y2), (255), 5) 
							

					cv2.imwrite(outputfolder + "/sdmap_%d.jpg" % (counter_out), sdmap)
					counter_out += 1
					print(counter_out)
			
			#exit()

			counter += 1
			
print(total_length, total_length / 8 / 1000.0)


