import random
import math
import numpy as np
class Vertex:
    def __init__(self, key, compute_res):
        self.id = key
        self.connectedTo = {}
        self.resource = compute_res
        self.distance = 10000
        self.predecessor = None
        self.state = 'On'

    def __str__(self):
        return str(self.id) + ' has resource ' + str(self.resource) + ' , and connectedTo ' + \
               str([x.id for x in self.connectedTo]) +  ' with link info ' + str(self.connectedTo.values())

    def addNeighbour(self, nbr, bw, dt):
        self.connectedTo[nbr] = [bw, dt]

    def delNeighbor(self, nbr):
        if nbr in self.connectedTo:
            del self.connectedTo[nbr]

    def setState(self, newstate):
        self.state = newstate

    def getState(self):
        return self.state

    def getConnections(self):
        return list(self.connectedTo.keys())

    def getConnection(self):
        return self.connectedTo.keys()

    def setDistance(self, distance):
        self.distance = distance

    def getDistance(self):
        return self.distance

    def setPred(self, r):
        self.predecessor = r

    def getPred(self):
        return self.predecessor

    def getId(self):
        return self.id

    def getLinkInfor(self, nbr):
        return self.connectedTo[nbr]

    def getResource(self):
        return self.resource

    def getWeight(self, nbr):
        return self.connectedTo[nbr][1]

class Network:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key, compute_res):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key, compute_res)
        self.vertList[key] = newVertex
        return newVertex

    def delateVertex(self, key):
        if key in self.vertList:
            del self.vertList[key]
            self.numVertices = self.numVertices - 1
        for i in self.getVertics():
            self.vertList[i].delNeighbor(key)

    def changeResorce(self, key, ch_comp):
        self.vertList[key].resource = self.vertList[key].resource + ch_comp

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, bw, dt):
        if f in self.vertList and t in self.vertList:
            self.vertList[f].addNeighbour(t, bw, dt)
            self.vertList[t].addNeighbour(f, bw, dt)

    def delEdge(self, f, t):
        if t in self.vertList[f].getConnections() or f in self.vertList[t].getConnections():
            if f in self.getVertics() and t not in self.getVertics():
                self.vertList[f].delNeighbor(t)
            elif f not in self.getVertics() and t in self.getVertics():
                self.vertList[t].delNeighbor(f)
            elif f in self.getVertics() and t in self.getVertics():
                self.vertList[f].delNeighbor(t)
                self.vertList[t].delNeighbor(f)

    def changeEdge(self, f, t, ch_bw):
        self.vertList[f].getLinkInfor(t)[0] = self.vertList[f].getLinkInfor(t)[0] + ch_bw
        self.vertList[t].getLinkInfor(f)[0] = self.vertList[t].getLinkInfor(f)[0] + ch_bw

    def getVertics(self):
        Vertics = []
        for i in self.vertList.keys():
            Vertics.append(i)
        return Vertics

    def getDis(self):
        Vertics = []
        for i in self.vertList.keys():
            Vertics.append(i)
        Dis = []
        for ii in Vertics:
            Dis.append(self.vertList[ii].getDistance())
        return Dis

    def getNeighbors(self, vertex):
        Neighbors = self.vertList[vertex].getConnections()
        for i in range(len(Neighbors) - 1, -1, -1):
            if self.vertList[Neighbors[i]].getState() != 'On':
                del Neighbors[i]
        return Neighbors

    def __iter__(self):
        return iter(self.vertList.values())

def buildedNetwork(initial_compute, initial_bandwidth):
    Net = Network()
    Net.addVertex('No1', initial_compute)
    Net.addVertex('No2', initial_compute)
    Net.addVertex('No3', initial_compute)
    Net.addVertex('No4', initial_compute)
    Net.addVertex('No5', initial_compute)
    Net.addVertex('No6', initial_compute)
    Net.addVertex('No7', initial_compute)
    Net.addVertex('No8', initial_compute)

    Net.addEdge('No1', 'No2', initial_bandwidth, 4)
    Net.addEdge('No1', 'No5', initial_bandwidth, 3)
    Net.addEdge('No7', 'No4', initial_bandwidth, 4)
    Net.addEdge('No2', 'No3', initial_bandwidth, 5)
    Net.addEdge('No3', 'No8', initial_bandwidth, 5)
    Net.addEdge('No4', 'No5', initial_bandwidth, 4)
    Net.addEdge('No6', 'No7', initial_bandwidth, 3)
    Net.addEdge('No6', 'No8', initial_bandwidth, 4)
    return Net

def request(Vertex):
    ratio = [0.7, 1.2, 1.1, 1.3, 0.7, 0.9, 1.2, 0.9]
    total_num_UEs = [i * 100 for i in ratio]
    uRLLC_to_mMTC_partition =  [2.5 / 10, 3 / 10, 4 / 10, 5 / 10, 3 / 10, 3 / 10, 4 / 10, 4 / 10] # mMTC请求的数量大于uRLLC
    Vertex_num_uRLLC = math.ceil(total_num_UEs[Vertex] * uRLLC_to_mMTC_partition[Vertex])
    Vertex_num_mMTC = math.floor(total_num_UEs[Vertex] * (1 - uRLLC_to_mMTC_partition[Vertex]))

    uRLLC_size = []
    mMTC_size = []
    x = 0
    y = 0
    while x < Vertex_num_uRLLC:
        req_comp = random.randrange(25, 28)
        uRLLC_size.append(req_comp)
        x += 1
    while y < Vertex_num_mMTC:
        req_comp = random.randrange(10, 13)
        mMTC_size.append(req_comp)
        y += 1
    mMTC_sum_size = sum(mMTC_size)
    uRLLC_sum_size = sum(uRLLC_size)

    req = [Vertex, Vertex_num_mMTC, Vertex_num_uRLLC, mMTC_sum_size, uRLLC_sum_size]
    return req

def requests(agents):
    requestss = []
    i = 0
    while i < agents:
        r = request(i)
        requestss.append(r)
        i += 1
    return requestss

def SourceState(network):
    Compresource = []
    for i in ('No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'No7', 'No8'):
        tempvalue1 = network.vertList[i].resource
        Compresource.append(tempvalue1)
    return Compresource

def matrix_to_vector(matrix):
    lis = []
    for i in matrix:
        for ii in i:
            lis.append(ii)
    return lis

def BWDSState(network):
    BWresource = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]]
    DSresource = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]]
    List = ('No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'No7', 'No8')
    for i in List:
        ii = List.index(i)
        for j in List:
            jj = List.index(j)
            if j in network.vertList[i].getConnections():
                BWresource[jj][ii] = network.vertList[i].getLinkInfor(j)[0]
                BWresource[ii][jj] = BWresource[jj][ii]
                DSresource[jj][ii] = network.vertList[i].getLinkInfor(j)[1]
                DSresource[ii][jj] = DSresource[jj][ii]
    return [BWresource, DSresource]


import copy
import itertools
from GetRewardTime import *

def negtive(list):
    for i in range(len(list)):
        list[i] = -list[i]
    return list


def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def Vertex_State_Update(network, Vertices, dist_Resources):
    for i in Vertices:
        network.changeResorce(i, dist_Resources[Vertices.index(i)])


def Path_State_Update(network, Vertices):
    if len(Vertices) == 2:
        network.changeEdge(Vertices[0], Vertices[1], 1)
    elif len(Vertices) > 2:
        for i in Vertices:
            if Vertices.index(i) < len(Vertices) - 1:
                network.changeEdge(i, Vertices[Vertices.index(i) + 1], 1)


def Negtive_Path_State_Update(network, Vertices):
    if len(Vertices) == 2:
        network.changeEdge(Vertices[0], Vertices[1], -1)
    elif len(Vertices) > 2:
        for i in Vertices:
            if Vertices.index(i) < len(Vertices) - 1:
                network.changeEdge(i, Vertices[Vertices.index(i) + 1], -1)


def actioninNN(ac):
    def Search(alist, item):
        pos = 0
        found = False
        while not found and pos < len(alist):
            if alist[pos] == item:
                found = True
            else:
                pos += 1
        return pos

    c = list(itertools.product([1, 2], [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]))
    newc = []
    for i in c:
        x = list(i)
        newc.append(x)
    index = Search(ac, 1)
    action = newc[index]
    return action


def GETposition(position):
    List = ['No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'No7', 'No8']
    return List[position]


class stepgo:
    def __init__(self, initial_compute, initial_bandwidth):
        self.network = buildedNetwork(initial_compute, initial_bandwidth)
        self.bwresource = BWDSState(self.network)[0]  # 17*17
        self.dtresource = BWDSState(self.network)[1]  # 17*17
        self.compresource = SourceState(self.network)  # 1*17
        self.reward = 0
        self.requestinfo = None  # 1*5
        self.AfterRunninglist = {}
        self.AfterRunninglog = {}

        self.AfterDownpath = {}
        self.AfterDownlog = {}

    def pickState(self):
        req_infor = copy.deepcopy(self.requestinfo)
        return [self.compresource, req_infor]

    def NetStateUpdate(self, timer):
        emptydir = {}
        if self.AfterRunninglist != emptydir:
            current_minimum_timelog = min(self.AfterRunninglist.values())
            current_minimum_timelog_req = get_keys(self.AfterRunninglist,
                                                   current_minimum_timelog)
            if timer >= current_minimum_timelog:
                for i in current_minimum_timelog_req:  # timer1\timer2\3..
                    Vertices = self.AfterRunninglog[i][0]
                    dist_Resources = self.AfterRunninglog[i][1]
                    Vertex_State_Update(self.network, Vertices, dist_Resources)
                    del self.AfterRunninglist[i]
                    del self.AfterRunninglog[i]


        if self.AfterDownpath != emptydir:
            current_minimum_timelog = min(self.AfterDownpath.values())
            current_minimum_timelog_req = get_keys(self.AfterDownpath, current_minimum_timelog)
            if timer >= current_minimum_timelog:
                for i in current_minimum_timelog_req:
                    Vertices = self.AfterDownlog[i][0]
                    Path_State_Update(self.network, Vertices)
                    del self.AfterDownpath[i]
                    del self.AfterDownlog[i]

        self.bwresource = BWDSState(self.network)[0]  # 17*17
        self.dtresource = BWDSState(self.network)[1]  # 17*17
        self.compresource = SourceState(self.network)  # 1*17
        self.requestinfo = requests(8)

        bwinfor = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]]
        for i in range(len(self.bwresource)):
            for ii in range(len(self.bwresource[i])):
                if self.bwresource[i][ii] <= 1:
                    bwinfor[i][ii] = 0
                else:
                    bwinfor[i][ii] = 1


    def ActionToReward(self, ac, timer, Ver):
        action = actioninNN(ac)
        req_position = GETposition(Ver)
        Vertices = []
        dist_Resources = []
        time = 0

        agency = copy.deepcopy(self.network)

        for i in agency.getVertics():
            if agency.vertList[i].getResource() < 25:
                agency.vertList[i].setState('Off')

        if agency.vertList[req_position].getState() == 'Off' and agency.getNeighbors(
                req_position) == []:
            self.reward = 0

        elif agency.vertList[req_position].getState() == 'Off' and agency.getNeighbors(
                req_position) != []:
            usable_neighbors = agency.getNeighbors(req_position)
            neighbors_resource = []
            for i in usable_neighbors:
                neighbors_resource.append(agency.vertList[i].getResource())
            TheChosen = usable_neighbors[neighbors_resource.index(max(neighbors_resource))]
            TheChosen_amount = max(neighbors_resource)

            if action[1] / 2 < TheChosen_amount / 3:
                self.reward, time = getReward(self.requestinfo[Ver], action[1] / 2)
                Vertices.append(req_position)
                Vertices.append(TheChosen)
                dist_Resources.append(0)
                dist_Resources.append(action[1] / 2)
            else:
                self.reward, time = getReward(self.requestinfo[Ver], TheChosen_amount / 3)
                time = time + 1
                Vertices.append(req_position)
                Vertices.append(TheChosen)
                dist_Resources.append(0)
                dist_Resources.append(TheChosen_amount / 3)

        elif agency.vertList[req_position].getState() == 'On' and agency.getNeighbors(req_position) == []:  #
            left_resource = agency.vertList[req_position].getResource()
            if left_resource < action[1]:
                self.reward, time = getReward(self.requestinfo[Ver], left_resource)
            else:
                self.reward, time = getReward(self.requestinfo[Ver], action[1])
            Vertices.append(req_position)
            dist_Resources.append(action[1])

        elif agency.vertList[req_position].getState() == 'On' and agency.getNeighbors(req_position) != []:
            if action[0] == 1:
                left_resource = agency.vertList[req_position].getResource()
                if left_resource < action[1]:
                    self.reward, time = getReward(self.requestinfo[Ver], left_resource)
                else:
                    self.reward, time = getReward(self.requestinfo[Ver], action[1])
                Vertices.append(req_position)
                dist_Resources.append(action[1])

            elif action[0] == 2:
                left_resource = agency.vertList[req_position].getResource()
                usable_neighbors = agency.getNeighbors(req_position)
                neighbors_resource = []
                for i in usable_neighbors:
                    neighbors_resource.append(agency.vertList[i].getResource())
                TheChosen = usable_neighbors[neighbors_resource.index(max(neighbors_resource))]
                TheChosen_Amount = max(neighbors_resource)
                if left_resource <= action[1] and TheChosen_Amount / 3 <= action[1] / 2:
                    self.reward, time = getReward(self.requestinfo[Ver], left_resource + TheChosen_Amount / 3)
                    Vertices.append(req_position)
                    Vertices.append(TheChosen)
                    dist_Resources.append(left_resource)
                    dist_Resources.append(TheChosen_Amount / 3)
                elif left_resource < action[1] and TheChosen_Amount / 3 > action[1] / 2:
                    self.reward, time = getReward(self.requestinfo[Ver], left_resource + action[1] / 2)
                    Vertices.append(req_position)
                    Vertices.append(TheChosen)
                    dist_Resources.append(left_resource)
                    dist_Resources.append(action[1] / 2)
                elif left_resource > action[1] and TheChosen_Amount / 3 < action[1] / 2:
                    self.reward, time = getReward(self.requestinfo[Ver], action[1] + TheChosen_Amount / 3)
                    Vertices.append(req_position)
                    Vertices.append(TheChosen)
                    dist_Resources.append(action[1])
                    dist_Resources.append(TheChosen_Amount / 3)
                elif left_resource >= action[1] and TheChosen_Amount / 3 >= action[1] / 2:
                    self.reward, time = getReward(self.requestinfo[Ver], action[1] + action[1] / 2)
                    Vertices.append(req_position)
                    Vertices.append(TheChosen)
                    dist_Resources.append(action[1])
                    dist_Resources.append(action[1] / 2)

        self.AfterRunninglist.update({str(timer): math.ceil(timer + (7-Ver) +time + sum(dist_Resources)/150)})
        self.AfterRunninglog.update({str(timer): [Vertices, dist_Resources]})


        self.AfterDownpath.update({str(timer): math.ceil(timer + (7-Ver) +time+ sum(dist_Resources)/150)})
        self.AfterDownlog.update({str(timer): [Vertices, dist_Resources]})
        dist_Resources = negtive(dist_Resources)
        Vertex_State_Update(self.network, Vertices, dist_Resources)
        Negtive_Path_State_Update(self.network, Vertices)
        negtive(dist_Resources)
        self.bwresource = BWDSState(self.network)[0]  # 17*17
        self.dtresource = BWDSState(self.network)[1]  # 17*17
        self.compresource = SourceState(self.network)

        req_infor = copy.deepcopy(self.requestinfo)

        bwinfor = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]]
        for i in range(len(self.bwresource)):
            for ii in range(len(self.bwresource[i])):
                if self.bwresource[i][ii] <= 1:
                    bwinfor[i][ii] = 0
                else:
                    bwinfor[i][ii] = 1

        state = [self.compresource, req_infor]
        return state, self.reward

    def getuniobs(self, computing, agents_request):
        # computing = list(np.divide(computing, 251))
        obs = [[],[],[],[],[],[],[],[]]
        List = ['No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'No7', 'No8']
        for i in range(len(List)):
            obs[i].append(computing[i])
            neighs = self.network.getNeighbors(List[i])
            for ii in neighs:
                n = computing[List.index(ii)]
                obs[i].append(n)
        iz = 0

        while iz < len(obs):
            for iii in agents_request[iz]:
                obs[iz].append(iii)
            obs[iz] = obs[iz][0:3] + obs[iz][4:]
            iz += 1
        return obs

    def uniform(self, state):
        for i in range(len(state[0])):
            state[0][i] = state[0][i] / 301
        for i in state[1]:
            i[0] = i[0] / 8
            i[1] = i[1] / 100
            i[2] = i[2] / 100
            i[3] = i[3] / 1500
            i[4] = i[4] / 1500
        unistate = []
        for ii in state[0]:
            unistate.append(ii)
        for iii in state[1]:
            for iiii in iii:
                unistate.append(iiii)
        return unistate

