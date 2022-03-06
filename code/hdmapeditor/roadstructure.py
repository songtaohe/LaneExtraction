import pickle
ENABLE_LOG = False

def dist2(p1,p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return a*a + b*b

class LaneMap():
    def __init__(self):
        self.nodes = {}
        self.nid = 0 

        self.neighbors = {}
        self.neighbors_all = {}

        self.edgeType = {} # "way" or "link"
        self.nodeType = {}

        self.history = []

    def updateNodeType(self):
        for nid, pos in self.nodes.items():
            nei = self.neighbors_all[nid]
            if len(nei) == 0:
                self.nodeType[nid] = "way"
            else:
                allLink = True
                for nn in nei:
                    edge = (nid, nn)
                    if self.edgeType[edge] == "way":
                        allLink = False
                        break
                if allLink:
                    self.nodeType[nid] = "link"
                else:
                    self.nodeType[nid] = "way"

        # create index?

    def query(self, p, nodelist = None):
        if nodelist is None or len(nodelist) == 0:
            bestd, bestnid = 10*10, None
            for nid, pos in self.nodes.items():
                if self.nodeType[nid] == "link":
                    continue

                d = dist2(p, pos)
                if d < bestd:
                    bestd = d 
                    bestnid = nid

            return bestnid

        else:
            bestd, bestnid = 10*10, None
            for nid in nodelist:
                pos = self.nodes[nid]
                # if self.nodeType[nid] == "link":
                #     continue

                d = dist2(p, pos)
                if d < bestd:
                    bestd = d 
                    bestnid = nid

            return bestnid
    
    def findLink(self, nid):
        nodelist = []
        queue = [nid]
        badnodes = []
        while len(queue) > 0:
            cur = queue.pop()
            for nn in self.neighbors_all[cur]:
                if nn not in self.nodeType:
                    print("Error cannot find node %d in nodeType" % nn)
                    if nn not in badnodes:
                        badnodes.append(nn)
                else:
                    if self.nodeType[nn] == "link":
                        if nn not in nodelist:
                            nodelist.append(nn)
                            queue.append(nn)
        for n in badnodes:
            self.deleteNode(n)

        return nodelist

    def findAllPolygons(self):
        visited = set()
        polygons = []
        for nid, _ in self.nodes.items():
            if nid in visited:
                continue

            start = nid 
            cur = nid 
            polygon = []
            while True:
                polygon.append(cur)
                nei = self.neighbors[cur]
                if len(nei) != 1:
                    break

                cur = nei[0]

                if cur == start:
                    polygon.append(start)
                    break
                if cur in polygon:
                    break

            if len(polygon) > 1 and polygon[0] == polygon[-1]:
                polygons.append(polygon)
           
        return polygons


    def addNode(self, p, updateNodeType = True):
        self.history.append(["addNode", p, self.nid])

        self.nodes[self.nid] = p
        self.neighbors[self.nid] = []
        self.neighbors_all[self.nid] = []
        self.nid += 1
        if updateNodeType:
            self.updateNodeType()
        return self.nid - 1

    def addEdge(self, n1, n2, edgetype = "way", updateNodeType = True):
        self.history.append(["addEdge", n1, n2, edgetype])

        if n2 not in  self.neighbors[n1]:
            self.neighbors[n1].append(n2)
        
        if n2 not in  self.neighbors_all[n1]:
            self.neighbors_all[n1].append(n2)
        
        if n1 not in  self.neighbors_all[n2]:
            self.neighbors_all[n2].append(n1)
        
        
        edge = (n1,n2)
        self.edgeType[edge] = edgetype
        edge = (n2,n1)
        self.edgeType[edge] = edgetype
        if updateNodeType:
            self.updateNodeType()

        pass
    
    def deleteNode(self, nid):
        self.history.append(["deleteNode", nid])
        if ENABLE_LOG:
            print("delete", nid)

        if nid in self.neighbors_all:
            neilist = list(self.neighbors_all[nid])
            for nn in neilist:
                self.deleteEdge(nn, nid)
                self.deleteEdge(nid, nn)

        if nid in self.nodes:
            del self.nodes[nid]
        
        if nid in self.neighbors:
            del self.neighbors[nid]

        if nid in self.neighbors_all:
            del self.neighbors_all[nid]

        if nid in self.nodeType:
            del self.nodeType[nid]

    def deleteEdge(self, n1, n2):
        self.history.append(["deleteEdge", n1, n2])
        if ENABLE_LOG:
            print("delete edge", n1, n2)
        if n1 in self.neighbors and n2 in self.neighbors[n1]:
            self.neighbors[n1].remove(n2)
            if ENABLE_LOG:
                print(self.neighbors[n1], n2)
        if n1 in self.neighbors_all and n2 in self.neighbors_all[n1]:
            self.neighbors_all[n1].remove(n2)
            if ENABLE_LOG:
                print(self.neighbors_all[n1], n2)
        if n2 in self.neighbors_all and n1 in self.neighbors_all[n2]:
            self.neighbors_all[n2].remove(n1)
            if ENABLE_LOG:
                print(self.neighbors_all[n2], n1)

        if (n1,n2) in self.edgeType:
            del self.edgeType[(n1,n2)]

    def checkConsistency(self):
        for nid in self.nodes.keys():
            if nid not in self.neighbors_all:
                self.neighbors_all[nid] = []
                print("missing neighbors_all", nid)

        for nid in self.nodes.keys():
            for nn in self.neighbors_all[nid]:
                if nn not in self.neighbors_all:
                    print("unsolved error", nn)
                    continue

                if nid not in self.neighbors_all[nn]:
                    print("incomplete neighbors", nid, nn)
                    self.neighbors_all[nn].append(nid)



    def undo(self):
        if len(self.history) > 0:
            item = self.history.pop()
            if item[0] == "addNode":
                nid = item[2]
                del self.nodes[nid]
                del self.neighbors[nid]
                del self.neighbors_all[nid]
                del self.nodeType[nid]
                #self.nid = nid
                # edgetype?

            
            elif item[0] == "addEdge":
                n1, n2, edgeType = item[1], item[2], item[3]
                if n2 in self.neighbors[n1]:
                    self.neighbors[n1].remove(n2)
                if n2 in self.neighbors_all[n1]:
                    self.neighbors_all[n1].remove(n2)
                if n1 in self.neighbors_all[n2]:
                    self.neighbors_all[n2].remove(n1)
                # edgetype?







    