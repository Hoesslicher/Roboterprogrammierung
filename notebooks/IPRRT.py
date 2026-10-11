# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from scipy.spatial import cKDTree
from IPPRMBase import PRMBase
import numpy as np

import networkx as nx
import random

from IPPerfMonitor import IPPerfMonitor

class RRTSimple(PRMBase):

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRTSimple, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
            
        Example:
            config["numberOfGeneratedNodes"] = 500 
            config["testGoalAfterNumberOfNodes"]  = 10
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0
        
        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)

        # 2. add start and goal to graph
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
        self.lastGeneratedNodeNumber +=1
        
        try:
            while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

                posList = list(nx.get_node_attributes(self.graph,'pos').values())
                kdTree = cKDTree(posList)

                if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
                    #print "testing goal"
                    result = kdTree.query(checkedGoalList[0],k=1)
                    if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],checkedGoalList[0]):
                        self.graph.add_node("goal", pos=checkedGoalList[0])
                        self.graph.add_edge(result[1],"goal")
                        mapping={0:'start'}
                        self.graph = nx.relabel_nodes(self.graph,mapping)

                        return nx.shortest_path(self.graph,"start","goal")


                pos = self._getRandomFreePosition()

                # for every node in graph find nearest neigbhours

                result = kdTree.query(pos,k=1)
                if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],pos):
                    self.graph.add_node(self.lastGeneratedNodeNumber, pos=pos)
                    self.graph.add_edge(result[1],self.lastGeneratedNodeNumber)
                    self.lastGeneratedNodeNumber +=1
        except Exception as e:
            print(e)

        return []


def nearestPointOfTree(graph, pos):
    """Nearest point of a tree (nodes and edges) to pos, all edges are searched.
    Returns (point, node, edge): node is the nearest node or None, if the point lies inside the edge."""
    edges = list(graph.edges())
    if not edges:
        # only the start
        node = next(iter(graph.nodes))
        return graph.nodes[node]['pos'], node, None
    p = np.asarray(pos, dtype=float)
    a = np.array([graph.nodes[u]['pos'] for u, v in edges], dtype=float)
    b = np.array([graph.nodes[v]['pos'] for u, v in edges], dtype=float)
    ab = b - a
    # projection of pos onto every edge: a + t * (b - a), t limited to [0, 1]
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0.0, 1.0)
    points = a + t[:, None] * ab
    k = int(np.argmin(np.linalg.norm(points - p, axis=1)))
    u, v = edges[k]
    if t[k] <= 1e-9:
        return graph.nodes[u]['pos'], u, None
    if t[k] >= 1.0 - 1e-9:
        return graph.nodes[v]['pos'], v, None
    return [float(x) for x in points[k]], None, (u, v)


def splitEdge(graph, point, edge, name):
    """Insert the node name at point into the edge (u, v): (u, name) and (name, v) replace (u, v)"""
    u, v = edge
    graph.add_node(name, pos=point)
    graph.remove_edge(u, v)
    graph.add_edge(u, name)
    graph.add_edge(name, v)


class RRTEdges(PRMBase):
    """RRTSimple, but q_near is the nearest point of the whole tree, i.e. of its nodes and edges
    (nearest(S(G), alpha(i)) on the slides). If it lies inside an edge, the edge is split there by a new node.
    """

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRTEdges, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0
        self.splitNodes = []    # nodes inserted into edges

    def _nearestPoint(self, pos):
        return nearestPointOfTree(self.graph, pos)

    def _insertNode(self, point, node, edge):
        """Node at the nearest point: the node itself or a new node, which splits the edge"""
        if node is not None:
            return node
        splitNode = "split" + str(len(self.splitNodes))
        splitEdge(self.graph, point, edge, splitNode)
        self.splitNodes.append(splitNode)
        return splitNode

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:
            config["numberOfGeneratedNodes"] = 500
            config["testGoalAfterNumberOfNodes"]  = 10

        Only the generated nodes (start and the added random configurations) are counted, the nodes inserted into
        edges (self.splitNodes) come on top.
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0
        self.splitNodes = []

        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        goal = checkedGoalList[0]

        # 2. add start to graph
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
        self.lastGeneratedNodeNumber += 1

        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

            if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
                point, node, edge = self._nearestPoint(goal)
                if not self._collisionChecker.lineInCollision(point, goal):
                    near = self._insertNode(point, node, edge)
                    self.graph.add_node("goal", pos=goal)
                    self.graph.add_edge(near, "goal")
                    self.graph = nx.relabel_nodes(self.graph, {0: 'start'})
                    return nx.shortest_path(self.graph, "start", "goal")

            pos = self._getRandomFreePosition()

            # nearest point of the tree: a node or a point inside an edge
            point, node, edge = self._nearestPoint(pos)
            if not self._collisionChecker.lineInCollision(point, pos):
                near = self._insertNode(point, node, edge)
                self.graph.add_node(self.lastGeneratedNodeNumber, pos=pos)
                self.graph.add_edge(near, self.lastGeneratedNodeNumber)
                self.lastGeneratedNodeNumber += 1

        return []


### following code is not yet finalized or corrected
            

from scipy.spatial import cKDTree
from IPPRMBase import PRMBase
import numpy as np

import networkx as nx
import random
class RRT(PRMBase):

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRT, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0


    def planPath(self, startList, goalList, config):
        """
        
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
            
        Example:
            config["numberOfGeneratedNodes"] = 500 
            config["testGoalAfterNumberOfNodes"]  = 10
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0
        
        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)
        
        # 2. add start and goal to graph
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
        self.lastGeneratedNodeNumber +=1
        
        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:
                    
            posList = list(nx.get_node_attributes(self.graph,'pos').values())
            kdTree = cKDTree(posList)

            if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
                #print "testing goal"
                result = kdTree.query(checkedGoalList[0],k=1)
                if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],checkedGoalList[0]):
                    self.graph.add_node("goal", pos=checkedGoalList[0])
                    self.graph.add_edge(result[1],"goal")
                    mapping={0:'start'}
                    self.graph = nx.relabel_nodes(self.graph,mapping)

                    return nx.shortest_path(self.graph,"start","goal")

            

            pos = self._getRandomFreePosition()
            
            # for every node in graph find nearest neigbhours
            result = kdTree.query(pos,k=1)

            start = np.array(self.graph.nodes[result[1]]['pos'])
            end   = np.array(pos)
            newPos = 0.5 * (end-start) + start
            if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],newPos):
                self.graph.add_node(self.lastGeneratedNodeNumber, pos=newPos)
                self.graph.add_edge(result[1],self.lastGeneratedNodeNumber)
                self.lastGeneratedNodeNumber +=1
            
 