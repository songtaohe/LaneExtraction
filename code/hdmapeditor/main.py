import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import cv2
import json
import sys 
import os 
import numpy as np 
from time import time, sleep
from roadstructure import LaneMap 

import math 
import pickle
import requests

LITE_RENDER = False

image = cv2.imread(sys.argv[1])

if len(sys.argv) > 4:
	b,g,r = np.copy(image[:,:,0]), np.copy(image[:,:,1]), np.copy(image[:,:,2])
	image[:,:,0] = r
	image[:,:,1] = b
	image[:,:,2] = g
	

margin = 128
margin_for_cnn = 512

image = np.pad(image, [[margin, margin], [margin, margin],[0,0]], 'constant')
image = image # // 4 * 3
image_for_cnn = np.pad(image, [[margin_for_cnn-margin, margin_for_cnn-margin], [margin_for_cnn-margin, margin_for_cnn-margin],[0,0]], 'constant')

dim = np.shape(image)
imageDim = dim
#windowsize = [1800,960]
#windowsize = [1560,900]
#windowsize = [1440,800]
windowsize = [1440,1200]
pos = [0,0]
frame = image[pos[1]:pos[1] + windowsize[1], pos[0]:pos[0] + windowsize[0], :]
minimap = cv2.resize(image, (256,256))
renderTime = 0

vis_switch_no_direction = False
vis_switch_no_arrow = False
vis_switch_no_minimap = False
vis_switch_no_diff_way_and_link = False
vis_switch_no_vertices = False
vis_switch_no_way = False
vis_switch_no_link = False





if len(sys.argv) > 2:
	annotation = sys.argv[2]
else:
	if ".jpg" in sys.argv[1]:
		annotation = sys.argv[1].replace(".jpg", "_label.p")
	else:
		annotation = sys.argv[1].replace(".png", "_label.p")

if os.path.exists(annotation):
	data = pickle.load(open(annotation, "rb"))
	if type(data) == list:
		laneMaps = data
	else:
		laneMaps = [data, LaneMap()]
else:
	laneMaps = [LaneMap(), LaneMap()]	
#laneMaps[1] = LaneMap()
laneMap = laneMaps[0]
laneMap.checkConsistency()
# for _ in range(17):
# 	laneMap.undo()

activeLaneMap = 0

step = 0

# some local states
lastMousex, lastMousey = -1, -1
mousex, mousey = 0, 0


lastNodeID = None


deleteMode = "(delete)"
deleteMode = " "

editingMode = "autofill_stage1"
editingMode = "autofill_stage2"
autofill_mode = "ml"
autofill_mode = "bezier"


autofill_nodes = [-1,-1]

editingMode = "erase"
erase_size = 1
editingMode = "ready_to_draw"
editingMode = "drawing_polyline"
editingMode = "ready_to_edit"
editingMode = "editing"
editingMode = "selecting_link"

edgeType = "link"
edgeType = "way"

activeLinks = []

editingMode = "ready_to_draw"

zoom = 1











def mouseEventHandler(event,x,y,flags,param):
	global mousex, mousey
	global editingMode
	global edgeType
	global lastNodeID
	global activeLinks
	global zoom 
	global laneMap
	global erase_size
	mousex, mousey = x, y
	
	#print(x,y,event, flags, param)
	global_x, global_y = x // zoom + pos[0], y // zoom + pos[1]

	if event == cv2.EVENT_LBUTTONUP:
		# if click an empty position
		#    start drawing lines
		# if click an existing node
		#    (0) if it is the last node then stop drawing
		#    (1) moving it
		#    (2) extend from it (default)

		if editingMode == "ready_to_draw":
			existing_node = laneMap.query((global_x, global_y), activeLinks)
			if existing_node is None:
				lastNodeID = laneMap.addNode((global_x, global_y))
				editingMode = "drawing_polyline"

			else:
				lastNodeID = existing_node
				editingMode = "drawing_polyline"
			

		elif editingMode == "drawing_polyline":
			existing_node = laneMap.query((global_x, global_y), activeLinks)
			if existing_node is None:
				# if in delete mode, do nothing
				if deleteMode == "(delete)":
					pass 
				else:
					nid = laneMap.addNode((global_x, global_y))
					# if nid == lastNodeID: # stop drawing 
					#     editingMode == "ready_to_draw":
					#     lastNodeID = None 
					# else:
					laneMap.addEdge(lastNodeID, nid, edgeType)
					lastNodeID = nid 
				
			else:
				if existing_node == lastNodeID: # stop drawing 
					# if in delete mode, delete the node
					if deleteMode == "(delete)":
						laneMap.deleteNode(lastNodeID)
						if lastNodeID in activeLinks:
							activeLinks.remove(lastNodeID)
					
					editingMode ="ready_to_draw"
					lastNodeID = None
				else:
					# if in delete mode, delete the edge
					if deleteMode == "(delete)":
						laneMap.deleteEdge(lastNodeID, existing_node)
					else:
						laneMap.addEdge(lastNodeID, existing_node, edgeType)
					lastNodeID = None
					editingMode = "ready_to_draw"

		elif editingMode == "ready_to_edit":
			existing_node = laneMap.query((global_x, global_y), activeLinks)
			if existing_node is not None:
				lastNodeID = existing_node
				editingMode = "editing"
		
		elif editingMode == "editing":
			editingMode = "ready_to_edit"
			lastNodeID = None 

		elif editingMode == "selecting_link":
			
			existing_node = laneMap.query((global_x, global_y))
			if existing_node is not None:
				activeLinks = laneMap.findLink(existing_node)
				#if len(activeLinks) > 0:
			else:
				activeLinks = []

			editingMode = "ready_to_edit"
		
		elif editingMode == "autofill_stage1":
			existing_node = laneMap.query((global_x, global_y))
			if existing_node is not None:
				autofill_nodes[0] = existing_node
				editingMode = "autofill_stage2"
				lastNodeID = existing_node


		elif editingMode == "autofill_stage2":
			existing_node = laneMap.query((global_x, global_y))
			if existing_node is not None:
				autofill_nodes[1] = existing_node

				if autofill_mode == "ml":
					# create input data
					r1 = laneMap.nodes[autofill_nodes[0]][1] + margin_for_cnn - margin
					c1 = laneMap.nodes[autofill_nodes[0]][0] + margin_for_cnn - margin
					r2 = laneMap.nodes[autofill_nodes[1]][1] + margin_for_cnn - margin
					c2 = laneMap.nodes[autofill_nodes[1]][0] + margin_for_cnn - margin
					
					mr = (r1+r2) // 2
					mc = (c1+c2) // 2
					cnnsize = 640

					sat = image_for_cnn[mr-cnnsize//2:mr+cnnsize//2,mc-cnnsize//2:mc+cnnsize//2,:]

					# render 
					seg = np.zeros_like(sat)
					direction = np.zeros_like(sat) + 127
					connector = np.zeros_like(sat)
					
					# 
					for nid, nei in laneMap.neighbors.items():
						x1 = laneMap.nodes[nid][0] + margin_for_cnn - margin - (mc-cnnsize//2)
						y1 = laneMap.nodes[nid][1] + margin_for_cnn - margin - (mr-cnnsize//2)
						
						for nn in nei:
							if laneMap.edgeType[(nid,nn)] != "way":
								continue
							
							x2 = laneMap.nodes[nn][0] + margin_for_cnn - margin - (mc-cnnsize//2)
							y2 = laneMap.nodes[nn][1] + margin_for_cnn - margin - (mr-cnnsize//2)

							dx = x2 - x1
							dy = y2 - y1
							l = math.sqrt(float(dx*dx + dy*dy))
							dx /= l 
							dy /= l

							if laneMap.edgeType[(nid,nn)] == "way":
								#color = (127 + int(dx * 127), 127 + int(dy * 127), 127)
								color = (127 + int(dx * 127), 127 + int(dy * 127), 127)
							
							cv2.line(seg, (x1,y1), (x2,y2), (255,255,255), 5)
							cv2.line(direction, (x1,y1), (x2,y2), color, 5)
							
					cv2.circle(connector, (c1 - (mc-cnnsize//2), r1 - (mr-cnnsize//2)), 8, (255,255,255), -1)
					cv2.circle(connector, (c2 - (mc-cnnsize//2), r2 - (mr-cnnsize//2)), 8, (255,255,255), -1)

					cv2.imwrite("tmp_sat.png", sat)
					cv2.imwrite("tmp_direction.png", direction)
					cv2.imwrite("tmp_connector.png", connector)
					cv2.imwrite("tmp_seg.png", seg)

					
					# TODO connect to server
					data = {"p1": [c1 - (mc-cnnsize//2), r1 - (mr-cnnsize//2)]}
					data["p2"] = [c2 - (mc-cnnsize//2), r2 - (mr-cnnsize//2)]
					data["img_in"] = "../hdmapeditor/tmp"

					r = requests.post('http://localhost:8080/',data = json.dumps(data))
					ret = json.loads(r.text)
					print(ret)
					if ret["success"] == "success":
						link = ret["result"][2]
						lastnid = autofill_nodes[0]
						for i in range(1, len(link)):
							x1 = link[i][1] + (mc-cnnsize//2) + margin - margin_for_cnn
							y1 = link[i][0] + (mr-cnnsize//2) + margin - margin_for_cnn
							
							nid1 = lastnid 

							if i == len(link)-1:
								nid2 = autofill_nodes[1]
							else:
								nid2 = laneMap.addNode((x1,y1))
								lastnid = nid2 

							laneMap.addEdge(nid1, nid2, edgetype="link")
					editingMode = "autofill_stage1"
					lastNodeID = None

				elif autofill_mode == "bezier":
					#p1 = laneMap.nodes[autofill_nodes[0]]
					#p2 = laneMap.nodes[autofill_nodes[1]]

					editingMode = "autofill_stage3"
					lastNodeID = None

		elif editingMode == "autofill_stage3":
			#global_x, global_y
			# create bezier curver
			x1,y1 = laneMap.nodes[autofill_nodes[0]]
			x2,y2 = laneMap.nodes[autofill_nodes[1]]
			x3,y3 = global_x, global_y

			L = np.sqrt((x1-x2)**2 + (y1-y2)**2)
			N = int(L / 20)+1
			def interpolate(p1,p2,a):
				return (p1[0] * (1-a) + p2[0] * a, p1[1] * (1-a) + p2[1] * a)

			prev_loc = (x1,y1)
			prev_nid = autofill_nodes[0]
			for i in range(N):
				alpha = float(i+1) / N 
				loc = interpolate(interpolate((x1,y1), (x3,y3), alpha),interpolate((x3,y3), (x2,y2), alpha),alpha) 

				if i == N - 1:
					nid = autofill_nodes[1]
				else:
					nid = laneMap.addNode((int(loc[0]),int(loc[1])))
				
				laneMap.addEdge(prev_nid, nid, edgetype="link")
				
				#cv2.line(frame, (int(prev_loc[0]),int(prev_loc[1])), (int(loc[0]),int(loc[1])), (255,0,255),2,cv2.LINE_AA)
				prev_loc = loc 
				prev_nid = nid 


			editingMode = "autofill_stage1"

				
		elif editingMode == "erase":
			rmlist = []
			for nid, loc in laneMap.nodes.items():
				if loc[0] > global_x - erase_size*50//zoom and loc[0] < global_x + erase_size*50//zoom:
					if loc[1] > global_y - erase_size*50//zoom and loc[1] < global_y + erase_size*50//zoom: 
						rmlist.append(nid)

			for nid in rmlist:
				laneMap.deleteNode(nid)


		pickle.dump(laneMaps, open(annotation, "wb"))

	if editingMode == "editing" :
		laneMap.nodes[lastNodeID] = [global_x, global_y]
		


	redraw()

def dashline(img, p1,p2,color,width,linetype):
	L = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
	w = 8
	for i in range(-w//2,int(L),w*2):
		a1 = max(0, float(i) / L)
		a2 = float(i+w) / L

		x1 = int(p1[0] * (1-a1) + p2[0] * a1)
		y1 = int(p1[1] * (1-a1) + p2[1] * a1)
		 
		x2 = int(p1[0] * (1-a2) + p2[0] * a2)
		y2 = int(p1[1] * (1-a2) + p2[1] * a2)

		cv2.line(img, (x1,y1), (x2,y2), color,width,linetype)

		

def redraw(noshow=False, transpose = False):
	global zoom
	global activeLinks
	global laneMap
	global activeLaneMap
	global lastNodeID
	global renderTime
	global erase_size

	global vis_switch_no_direction
	global vis_switch_no_minimap
	global vis_switch_no_diff_way_and_link
	global vis_switch_no_vertices
	global vis_switch_no_way
	global vis_switch_no_link


	t0 = time()
	
	laneMap = laneMaps[0]
	currentLastNodeID = lastNodeID
	if activeLaneMap == 1:
		lastNodeID = None

	frame = np.copy(image[pos[1]:pos[1] + windowsize[1] // zoom, pos[0]:pos[0] + windowsize[0] // zoom, :])
	if zoom > 1:
		frame = cv2.resize(frame, (windowsize[0], windowsize[1]))
		
	#print(laneMap.neighbors)
	for renderpass in [0,1]:
		for nid, nei in laneMap.neighbors.items():
			x1 = (laneMap.nodes[nid][0] - pos[0]) * zoom
			y1 = (laneMap.nodes[nid][1] - pos[1]) * zoom
			outrange = False
			if (x1 < 0 or x1 > windowsize[0]) or (y1 < 0 or y1 > windowsize[1]):
				outrange = True
				#continue

			for nn in nei:
				
				if laneMap.edgeType[(nid,nn)] == "way":
					color = (0,255,0)
					lanetype = "way"
				else:
					if nid in activeLinks or nn in activeLinks:
						color = (230,230,0)
					else:
						color = (255,0,0)

					lanetype = "link"

				x2 = (laneMap.nodes[nn][0] - pos[0]) * zoom
				y2 = (laneMap.nodes[nn][1] - pos[1]) * zoom

				if (x2 < 0 or x2 > windowsize[0]) or (y2 < 0 or y2 > windowsize[1]):
					if outrange:
						continue

				# if LITE_RENDER:
				# 	cv2.line(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)
				# else:

				dx = x2 - x1
				dy = y2 - y1
				l = math.sqrt(float(dx*dx + dy*dy)) + 0.001
				dx /= l 
				dy /= l

				if vis_switch_no_direction:
					if laneMap.edgeType[(nid,nn)] == "way":
						dx = 1
						dy = 0
					else:
						dx = 0
						dy = 1
				

				if laneMap.edgeType[(nid,nn)] == "way":
					#color = (127, 127 + int(dx * 127), 127 + int(dy * 127))
					if transpose:
						color = (192, 192 + int(-dy * 63), 192 + int(-dx * 63))
					else:	
						color = (192, 192 + int(dx * 63), 192 + int(dy * 63))
				else:
					if transpose:
						color = (127, 127 + int(-dy * 127), 127 + int(-dx * 127))
					else:
						color = (127, 127 + int(dx * 127), 127 + int(dy * 127))

				if vis_switch_no_diff_way_and_link:
					color = (192,255,192)

				if vis_switch_no_link and laneMap.edgeType[(nid,nn)] == "link":
					continue
				
				if vis_switch_no_way and laneMap.edgeType[(nid,nn)] == "way":
					continue
			


				scale = 6

				mx = (x1+x2) // 2
				my = (y1+y2) // 2

				ms = []
				arrow_int = 40
				if l > arrow_int:
					N = int(l / arrow_int)
					for k in range(N+1):
						a = float(k+1) / (N+1)
						mx = int(x1 * (1-a) + x2 * a)
						my = int(y1 * (1-a) + y2 * a)
						ms.append((mx,my))


				else:
					ms = [(mx,my)]

				shadow_color = (96,96,96)
				shadow_width = 4
				line_width = 2

				if lanetype == "link":
					if LITE_RENDER or vis_switch_no_direction:
						dashline(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)
					else:
						if renderpass == 0:
							for mx, my in ms:
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx - int(dy * scale),my+int(dx * scale)), shadow_color,shadow_width,cv2.LINE_AA)
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx + int(dy * scale),my-int(dx * scale)), shadow_color,shadow_width,cv2.LINE_AA)
							dashline(frame, (x1,y1), (x2,y2), shadow_color,shadow_width,cv2.LINE_AA)
						else:
							for mx, my in ms:
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx - int(dy * scale),my+int(dx * scale)), color,2)
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx + int(dy * scale),my-int(dx * scale)), color,2)
							dashline(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)
				else:
					if LITE_RENDER or vis_switch_no_direction:
						cv2.line(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)
					else:
						if renderpass == 0:
							for mx, my in ms:
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx - int(dy * scale),my+int(dx * scale)), shadow_color,shadow_width,cv2.LINE_AA)
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx + int(dy * scale),my-int(dx * scale)), shadow_color,shadow_width,cv2.LINE_AA)
							cv2.line(frame, (x1,y1), (x2,y2), shadow_color,shadow_width,cv2.LINE_AA)
						else:
							for mx, my in ms:
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx - int(dy * scale),my+int(dx * scale)), color,2)
								if not vis_switch_no_arrow : cv2.line(frame, (mx + int(dx * scale),my+int(dy * scale)), (mx + int(dy * scale),my-int(dx * scale)), color,2)
							cv2.line(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)


	# 
		
	#if LITE_RENDER == False:
	if vis_switch_no_vertices == False:
		for nid, p in laneMap.nodes.items():
			x1 = (p[0] - pos[0]) * zoom
			y1 = (p[1] - pos[1]) * zoom

			if laneMap.nodeType[nid] == "way" and vis_switch_no_way == False:
				cv2.circle(frame, (x1,y1),4,(0,0,0),-1)
				#cv2.circle(frame, (x1,y1),3,(0,0,255),-1)
				
			elif laneMap.nodeType[nid] == "link" and vis_switch_no_link == False:
				cv2.circle(frame, (x1,y1),3,(255,0,0),-1)

	if vis_switch_no_way == False:
		for nid, p in laneMap.nodes.items():
			x1 = (p[0] - pos[0]) * zoom
			y1 = (p[1] - pos[1]) * zoom

			if laneMap.nodeType[nid] == "way":
				for nei in laneMap.neighbors_all[nid]:
					if ((nei, nid) in laneMap.edgeType and laneMap.edgeType[(nei, nid)] == "link") or ((nid, nei) in laneMap.edgeType and laneMap.edgeType[(nid, nei)] == "link"):
						cv2.circle(frame, (x1,y1),5,(255,0,0),2)
			

	# draw active nodes
	for nid in activeLinks:
		x1 = (laneMap.nodes[nid][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[nid][1] - pos[1]) * zoom
		cv2.circle(frame, (x1,y1),5,(255,255,0),5,1)

	# highlight snapped nodes
	global_x, global_y = mousex // zoom + pos[0], mousey // zoom + pos[1]
	if activeLaneMap == 0:
		existing_node = laneMap.query((global_x, global_y), activeLinks)
		if existing_node is not None:
			x1 = (laneMap.nodes[existing_node][0] - pos[0]) * zoom
			y1 = (laneMap.nodes[existing_node][1] - pos[1]) * zoom
			cv2.circle(frame, (x1,y1),5,(0,0,255),2)

	# draw virtual lines
	if (editingMode == "drawing_polyline" or editingMode == "autofill_stage2")  and lastNodeID is not None: 
		x1 = (laneMap.nodes[lastNodeID][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[lastNodeID][1] - pos[1]) * zoom
		x2 = mousex
		y2 = mousey 
		if edgeType == "way":
			cv2.line(frame, (x1,y1), (x2,y2), (0,255,0),2,cv2.LINE_AA)
		else:
			cv2.line(frame, (x1,y1), (x2,y2), (255,0,0),2,cv2.LINE_AA)
	
	if editingMode == "autofill_stage3":
		x1 = (laneMap.nodes[autofill_nodes[0]][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[autofill_nodes[0]][1] - pos[1]) * zoom
		x2 = (laneMap.nodes[autofill_nodes[1]][0] - pos[0]) * zoom
		y2 = (laneMap.nodes[autofill_nodes[1]][1] - pos[1]) * zoom
		x3 = mousex
		y3 = mousey 

		cv2.line(frame, (x1,y1), (x2,y2), (255,255,0),2,cv2.LINE_AA)
		cv2.line(frame, (x1,y1), (x3,y3), (255,255,0),1,cv2.LINE_AA)
		cv2.line(frame, (x2,y2), (x3,y3), (255,255,0),1,cv2.LINE_AA)

		# Bezier curver

		L = np.sqrt((x1-x2)**2 + (y1-y2)**2)
		N = int(L / 20)+1
		def interpolate(p1,p2,a):
			return (p1[0] * (1-a) + p2[0] * a, p1[1] * (1-a) + p2[1] * a)

		prev_loc = (x1,y1)
		for i in range(N):
			alpha = float(i+1) / N 
			loc = interpolate(interpolate((x1,y1), (x3,y3), alpha),interpolate((x3,y3), (x2,y2), alpha),alpha) 

			cv2.line(frame, (int(prev_loc[0]),int(prev_loc[1])), (int(loc[0]),int(loc[1])), (255,0,255),2,cv2.LINE_AA)
			prev_loc = loc 


	# render mask
	lastNodeID = currentLastNodeID
	if activeLaneMap == 0:
		lastNodeID = None

	laneMap = laneMaps[1]
	polygons = laneMap.findAllPolygons()
	
	mask = np.zeros_like(frame)

	for polygon in polygons:
		polygon_list = []
		for i in range(len(polygon)-1):
			x1 = (laneMap.nodes[polygon[i]][0] - pos[0]) * zoom
			y1 = (laneMap.nodes[polygon[i]][1] - pos[1]) * zoom
			x2 = (laneMap.nodes[polygon[i+1]][0] - pos[0]) * zoom
			y2 = (laneMap.nodes[polygon[i+1]][1] - pos[1]) * zoom
			if noshow == False:
				cv2.line(frame, (x1,y1), (x2,y2), (0,0,255),2,cv2.LINE_AA)

			polygon_list.append([x1,y1])
		polygon_list.append([x2,y2])

		area = np.array(polygon_list)
		area = area.astype(np.int32)
		if noshow == False:
			cv2.fillPoly(mask, [area], (50,50,255))

	for nid, nei in laneMap.neighbors.items():
		x1 = (laneMap.nodes[nid][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[nid][1] - pos[1]) * zoom
		
		for nn in nei:
			color = (0,0,255)


			x2 = (laneMap.nodes[nn][0] - pos[0]) * zoom
			y2 = (laneMap.nodes[nn][1] - pos[1]) * zoom

			dx = x2 - x1
			dy = y2 - y1
			l = math.sqrt(float(dx*dx + dy*dy))
			dx /= l 
			dy /= l
			scale = 5

			mx = (x1+x2) // 2
			my = (y1+y2) // 2
			if noshow == False:
				cv2.line(mask, (mx + int(dx * scale),my+int(dy * scale)), (mx - int(dy * scale),my+int(dx * scale)), color,2)
				cv2.line(mask, (mx + int(dx * scale),my+int(dy * scale)), (mx + int(dy * scale),my-int(dx * scale)), color,2)
			

				cv2.line(frame, (x1,y1), (x2,y2), color,2,cv2.LINE_AA)
	
	for nid, p in laneMap.nodes.items():
		x1 = (p[0] - pos[0]) * zoom
		y1 = (p[1] - pos[1]) * zoom

		if noshow == False:
			cv2.circle(mask, (x1,y1),3,(0,255,255),-1)
		
	# # draw active nodes
	# for nid in activeLinks:
	# 	x1 = (laneMap.nodes[nid][0] - pos[0]) * zoom
	# 	y1 = (laneMap.nodes[nid][1] - pos[1]) * zoom
	# 	cv2.circle(mask, (x1,y1),5,(255,255,0),5,1)

	# highlight snapped nodes
	global_x, global_y = mousex // zoom + pos[0], mousey // zoom + pos[1]
	existing_node = laneMap.query((global_x, global_y))
	if existing_node is not None:
		x1 = (laneMap.nodes[existing_node][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[existing_node][1] - pos[1]) * zoom
		cv2.circle(mask, (x1,y1),5,(0,0,255),2)

	# draw virtual lines
	if editingMode == "drawing_polyline" and lastNodeID is not None: 
		x1 = (laneMap.nodes[lastNodeID][0] - pos[0]) * zoom
		y1 = (laneMap.nodes[lastNodeID][1] - pos[1]) * zoom
		x2 = mousex
		y2 = mousey 
		if edgeType == "way":
			cv2.line(mask, (x1,y1), (x2,y2), (0,255,0),2,cv2.LINE_AA)
		else:
			cv2.line(mask, (x1,y1), (x2,y2), (255,0,0),2,cv2.LINE_AA)

	
	frame = cv2.add(frame, mask)
	lastNodeID = currentLastNodeID

	# copy to minimap
	crop = cv2.resize(frame, (int(float(windowsize[0] // zoom) / imageDim[0] * 256), int(float(windowsize[1] // zoom) / imageDim[1] * 256) ), interpolation = cv2.INTER_LANCZOS4)
	r1 = int(float(pos[1]) / imageDim[0] * 256)
	c1 = int(float(pos[0]) / imageDim[1] * 256)
	r2 = r1 + np.shape(crop)[0]
	c2 = c1 + np.shape(crop)[1]
	minimap[r1:r2,c1:c2,:] = crop	

	# draw minimap
	if vis_switch_no_minimap == False:
		frame[10:10+256, windowsize[0]-256-10:windowsize[0]-10,:] = minimap
		cv2.rectangle(frame, (windowsize[0]-256-10, 10+256), (windowsize[0]-10, 10), (255,255,255), 2)

		x1 = windowsize[0]-256-10 + int(float(pos[0]) / imageDim[0] * 256)
		x2 = windowsize[0]-256-10 + int(float(pos[0] + windowsize[0] // zoom) / imageDim[0] * 256)
		y1 = 10 + int(float(pos[1] + windowsize[1] // zoom) / imageDim[0] * 256)
		y2 = 10 + int(float(pos[1]) / imageDim[0] * 256)

		cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

	# draw crossing line
	color = (255,255,255)
	if editingMode == "autofill_stage1":
		color = (255,0,0)

	if editingMode == "ready_to_edit":
		color = (0,255,255)
	
	if deleteMode == "(delete)":
		color = (0,0,255)

	if editingMode == "erase":
		color = (0,0,255)
		cv2.rectangle(frame, (mousex-erase_size*50, mousey-erase_size*50), (mousex+erase_size*50, mousey+erase_size*50), color, 2)

	else:
		cv2.line(frame, (mousex - 50, mousey), (mousex + 50, mousey), color, 1)
		cv2.line(frame, (mousex, mousey - 50), (mousex, mousey + 50), color, 1)
	
	# add text
	if noshow == False:
		cv2.putText(frame, "%s | %s | Layer %d | Render Time %.3f Seconds" % (editingMode+deleteMode, edgeType, activeLaneMap, renderTime), (10,32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
				
	laneMap = laneMaps[activeLaneMap]
	if noshow==False:
		cv2.imshow("image",frame)
	renderTime = time() - t0

	return frame

if len(sys.argv) > 3:
	config = json.load(open(sys.argv[3]))

	pos = config["pos"]

	

	vis_switch_no_direction = config["no_direction"] if "no_direction" in config else vis_switch_no_direction
	vis_switch_no_arrow = config["no_arrow"] if "no_arrow" in config else vis_switch_no_arrow
	vis_switch_no_minimap = config["no_minimap"] if "no_minimap" in config else vis_switch_no_minimap
	vis_switch_no_diff_way_and_link = config["no_diff_way_and_link"] if "no_diff_way_and_link" in config else vis_switch_no_diff_way_and_link
	vis_switch_no_vertices = config["no_vertices"] if "no_vertices" in config else vis_switch_no_vertices
	vis_switch_no_way = config["no_way"] if "no_way" in config else vis_switch_no_way
	vis_switch_no_link = config["no_link"] if "no_link" in config else vis_switch_no_link

	# if not (vis_switch_no_way and vis_switch_no_link):
	# 	image = image // 4 * 3

	if config["bk"] == "white":
		image = image * 0 + 255
	if "transpose" in config:
		transpose = True
	else:
		transpose = False
	frame = redraw(noshow=True, transpose = transpose)
	crop = config["crop"]
	img = frame[crop[1]:crop[3], crop[0]:crop[2]]

	if "transpose" in config:
		img = np.transpose(img, axes = [1,0,2])


	dim = np.shape(img)
	dimx = dim[0]
	dimy = dim[1]

	# py = 80
	# img[dimx-py-1:dimx-py+1, dimy-100:dimy-20, :] = 0

	
	# for i in range(11):
	# 	if i == 0 or i == 10:
	# 		img[dimx-py-10:dimx-py+10, dimy-101+i*8:dimy-99+i*8, :] = 0
	# 	else:
	# 		img[dimx-py-6:dimx-py+1, dimy-101+i*8:dimy-99+i*8, :] = 0
	


	cv2.imwrite(config["output"], img)

	exit()


cv2.namedWindow("image")
cv2.moveWindow("image", 0,0)
cv2.setMouseCallback("image", mouseEventHandler)

while True:
	hasUpdate = False
	if step == 0:
		hasUpdate = True
	step += 1

	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break

	if k == ord("w"):
		pos[1] = max(0, pos[1] - 100)
		hasUpdate = True
		print(pos)

	if k == ord("a"):
		pos[0] = max(0, pos[0] - 100)
		hasUpdate = True
		print(pos)
	
	if k == ord("s"):
		pos[1] = min(dim[0] - windowsize[1] // zoom, pos[1] + 100)
		hasUpdate = True
		print(pos)

	if k == ord("d"):
		pos[0] = min(dim[1] - windowsize[0] // zoom, pos[0] + 100)
		hasUpdate = True
		print(pos)
	
	if k == ord("e"):
		if editingMode == "ready_to_edit":
			editingMode = "ready_to_draw"
		elif editingMode == "ready_to_draw":
			editingMode = "ready_to_edit"
		hasUpdate = True

	if k == ord("f"):
		if editingMode == "ready_to_draw":
			editingMode = "autofill_stage1"
		else:
			editingMode = "ready_to_draw"

		hasUpdate = True
		
	if k == ord("q"):
		if edgeType == "link":
			edgeType = "way"
		elif edgeType == "way":
			edgeType = "link"
		hasUpdate = True

	if k == ord("z"):
		pass
		# not working with deletion yet... TODO
		laneMap.undo()
		hasUpdate = True
		pickle.dump(laneMaps, open(annotation, "wb"))

	if k == ord("c"):
		activeLinks = []
		editingMode = "selecting_link"
		hasUpdate = True

	if k == ord("1"):
		zoom = 1
		hasUpdate = True

	if k == ord("2"):
		zoom = 2
		hasUpdate = True

	if k == ord("3"):
		zoom = 3
		hasUpdate = True
	
	if k == ord("m"):
		activeLaneMap = (activeLaneMap + 1) % 2
		laneMap = laneMaps[activeLaneMap]

		edgeType = "way"

		hasUpdate = True
	
	if k == ord("x"):
		if deleteMode == " ":
			deleteMode = "(delete)"
		else:
			deleteMode = " "
		hasUpdate = True
	
	if k == ord("r"):
		if editingMode == "erase":
			editingMode = "ready_to_draw"
		else:
			editingMode = "erase"
		hasUpdate = True

	if k == ord("4"):
		erase_size = 1
		hasUpdate = True

	if k == ord("5"):
		erase_size = 2
		hasUpdate = True

	if k == ord("6"):
		erase_size = 4
		hasUpdate = True
	
	#print(lastMousex, mousex, lastMousey, mousey)

	if lastMousex != mousex or lastMousey != mousey:
		lastMousex, lastMousey = mousex, mousey
		hasUpdate = True

	if hasUpdate:
		redraw()



cv2.destroyAllWindows()