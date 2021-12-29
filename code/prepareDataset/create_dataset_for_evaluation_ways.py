# 25.812813, -80.202372

import json 
import math 
from subprocess import Popen 

import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from satellite import mapbox as md
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

			# find all ways 
			# - find nodes that belong to terminals or way intersections.
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
				elif way_c > 2 and link_c == 0:
					terminal_nodes.append(nid)
				elif way_c == 1 and link_c == 0:
					terminal_nodes.append(nid)
					
			wayset = set()
			ways = []
			for nid in terminal_nodes:
				queue = [[nid, []]]
				while len(queue) > 0:
					cur, curlist = queue.pop()
					#print(cur, curlist)
					curlist.append(cur)
					if cur in terminal_nodes and len(curlist) > 1 :
						ways.append(list(curlist))
						wayset.add((curlist[0], curlist[-1]))
						continue

					if len(curlist) > 1 and cur in curlist[:-1]:
						print("bug?")
						ways.append(list(curlist))
						wayset.add((curlist[0], curlist[-1]))
						continue
 

					for nn in roadlabel.neighbors[cur]:
						if roadlabel.nodeType[nn] == "way":
							if (cur, nn) in roadlabel.edgeType:
								edgetype = roadlabel.edgeType[(cur, nn)]
							else:
								edgetype = roadlabel.edgeType[(nn, cur)]
							if edgetype == "way":
								queue.append([nn, list(curlist)])
						#print(nn, curlist, cur in terminal_nodes, nn in terminal_nodes)

			def distance(a,b):
				return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

			newways = []
			for way in ways:
				node_list = [(roadlabel.nodes[nid][0], roadlabel.nodes[nid][1]) for nid in way]
				density = 50
				pd = [0]
				for i in range(len(node_list)-1):
					pd.append(pd[-1] + distance(node_list[i], node_list[i+1]))
				
				interpolate_N = int(pd[-1]/density)

				last_loc = node_list[0]
				nway = [last_loc]

				for i in range(interpolate_N):
					int_d = pd[-1]/(interpolate_N+1)*(i+1)
					for j in range(len(node_list)-1):
						if pd[j] <= int_d and pd[j+1] > int_d:
							a = (int_d-pd[j]) / (pd[j+1] - pd[j])

							loc = ((1-a) * node_list[j][0] + a * node_list[j+1][0], (1-a) * node_list[j][1] + a * node_list[j+1][1])
							nway.append((int(loc[0]), int(loc[1])) )
							last_loc = loc 
				nway.append(node_list[-1])
				newways.append(nway)
			
			ways = newways

			# nidmap = {}
			# for item in wayset:
			# 	n1,n2 = item 
			# 	if n1 not in nidmap:
			# 		nidmap[n1] = [n2]
			# 	else:
			# 		nidmap[n1].append(n2)

			# 	if n2 not in nidmap:
			# 		nidmap[n2] = [n1]
			# 	else:
			# 		nidmap[n2].append(n1)

			# for nid in terminal_nodes:
			# 	if nid not in nidmap:
			# 		nidmap[nid] = []
	
			print("number of ways", len(ways))
			#exit()
			#polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			
			sr = 0
			sc = 0
			mask = cv2.imread(outputfolder + "/regionmask_%d.jpg" % (counter_out))
			margin = 128

			localways = []
			for way in ways:
				vertices = []
				outOfRange = False
				in_cc = 0
				out_cc = 0
				for loc in way:
					x = loc[0] - sc - margin
					y = loc[1] - sr - margin
					vertices.append([x,y])
						
					if x > 0 and x < 4096 and y > 0 and y < 4096 and mask[y,x,0] > 127:
						in_cc += 1
					else:
						out_cc += 1
						#outOfRange = True

				if in_cc >= out_cc * 2:
					localways.append(vertices)
			
			# localnodes = {}

			# for nid in terminal_nodes:
			# 	x = roadlabel.nodes[nid][0] - sc - margin
			# 	y = roadlabel.nodes[nid][1] - sr - margin
			# 	if x > 0 and x < 4096 - 0 and y > 0 and y < 4096 - 0 and mask[y,x,0] > 127:
			# 		localnodes[nid] = [x,y]
			print(len(ways), " --> ", len(localways))
			json.dump(localways, open(outputfolder + "/way_%d.json" % (counter_out), "w"), indent=2)

			counter_out += 1
			print(counter_out)

			
			
			
			
			counter += 1
			
print(total_length, total_length / 8 / 1000.0)


