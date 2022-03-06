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

			#img = cv2.imread(folder + "/sat_%d.jpg" % (counter))
			try:
				labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
			except:
				break
			
			roadlabel, masklabel = labels 

			# find all links
			# - find nodes that belong to both link edges and way edges.
			# - search for a path between pairs of them.
			# - store them into a data structure for future use. 
			
			terminal_nodes = []
			for nid in roadlabel.nodes.keys():
				way_c = 0
				link_c = 0
				for nn in roadlabel.neighbors_all[nid]:
					if (nid, nn) in roadlabel.edgeType:
						edgetype = roadlabel.edgeType[(nid, nn)]
					else:
						edgetype = roadlabel.edgeType[(nn, nid)]

					if edgetype == "way":
						way_c += 1
					else:
						link_c += 1

				if way_c > 0 and link_c > 0:
					terminal_nodes.append(nid)

			linkset = set()
			links = []
			for nid in terminal_nodes:
				link = [nid]
				queue = [[nid, [nid]]]
				while len(queue) > 0:
					cur, curlist = queue.pop()
					#print(cur, curlist)
					for nn in roadlabel.neighbors[cur]:
						if roadlabel.edgeType[(cur, nn)] == "way":
							continue
						
						if roadlabel.nodeType[nn] == "link":
							newlist = list(curlist)
							newlist.append(nn)
							queue.append([nn, newlist])
						else:
							newlist = list(curlist)
							newlist.append(nn)
							if (nid, nn) not in linkset:
								linkset.add((nid, nn))
							if len(newlist) > 1:
								links.append(list(newlist))

			nidmap = {}

			for item in linkset:
				n1,n2 = item 
				if n1 not in nidmap:
					nidmap[n1] = [n2]
				else:
					nidmap[n1].append(n2)

				if n2 not in nidmap:
					nidmap[n2] = [n1]
				else:
					nidmap[n2].append(n1)

			for nid in terminal_nodes:
				if nid not in nidmap:
					nidmap[nid] = []

				

				
			print("number of links", len(links))
			#exit()
			polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			for sr in [0, 1024, 2048]:
				for sc in [0, 1024, 2048]:
					mask = cv2.imread(outputfolder + "/regionmask_%d.jpg" % (counter_out))
					margin = 128

					locallinks = []
					for link in links:
						vertices = []
						outOfRange = False
						for nid in link:
							x = roadlabel.nodes[nid][0] - sc - margin
							y = roadlabel.nodes[nid][1] - sr - margin

							if x > 0 and x < 2048 and y > 0 and y < 2048 and mask[y,x,0] > 127:
								vertices.append([x,y])
								pass
							else:
								outOfRange = True

						if not outOfRange:
							locallinks.append(vertices)
					
					localnodes = {}

					for nid in terminal_nodes:
						x = roadlabel.nodes[nid][0] - sc - margin
						y = roadlabel.nodes[nid][1] - sr - margin
						if x > 0 and x < 2048 - 0 and y > 0 and y < 2048 - 0 and mask[y,x,0] > 127:
							localnodes[nid] = [x,y]

					json.dump([nidmap, localnodes, locallinks], open(outputfolder + "/link_%d.json" % (counter_out), "w"), indent=2)

					counter_out += 1
					print(counter_out)

			
			
			
			counter += 1
			
print(total_length, total_length / 8 / 1000.0)


