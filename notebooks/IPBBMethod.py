# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

BB-Method (Baginski 1998): "Blow up" the robot and "Bend" the path.

A path is deformed until it is collision-free. Configurations are rated by their (virtual) penetration depth,
which the collision checker provides (penetrationDepth): Q(q) = -penetrationDepth(q), free: Q = 0.
A segment is rated by its worst configuration q_min.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

import networkx as nx
import numpy as np

from IPPerfMonitor import IPPerfMonitor
from IPPlanerBase import PlanerBase


def pathLength(path):
    """Length of a path given as list of configurations."""
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def perpendicularDirections(qa, qb):
    """Unit vectors perpendicular to the segment qa -> qb.
    2D: the normal of the segment, nD: an orthonormal basis of the n-1 dimensional complement."""
    if len(qa) == 2:
        length = math.dist(qa, qb)
        return [np.array([-(qb[1] - qa[1]) / length, (qb[0] - qa[0]) / length])]
    v = np.asarray(qb, dtype=float) - np.asarray(qa, dtype=float)
    v /= np.linalg.norm(v)
    basis = [v]
    for e in np.eye(len(v)):
        w = e - sum(np.dot(e, b) * b for b in basis)
        if np.linalg.norm(w) > 1e-6:
            basis.append(w / np.linalg.norm(w))
    return basis[1:]


class BBMethod(PlanerBase):

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface, it has to provide penetrationDepth()
        """
        super(BBMethod, self).__init__(_collChecker)
        self.graph = nx.Graph()   # the (last) path, nodes "start", 1, 2, ..., "goal" with attribute 'pos'
        self.path = []            # the (last) path as list of configurations
        self.steps = []           # protocol of every step, see _step()
        self._segmentCache = dict()

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:
            config["fStep"] = 0.8          # shifting displacement d = fStep * length of the segment ...
            config["deltaMin"] = 0.2       # ... limited to [deltaMin, deltaMax] (delta_1, delta_2 of the slides)
            config["deltaMax"] = 2.0
            config["fDivide"] = 0.3        # partitioning factor in [0, 1]
            config["resolution"] = 0.05    # (maximal) distance of the rated configurations on a segment
            config["maxIterations"] = 100

        Returns the path as list of nodes of self.graph ([] if no collision-free path was found).
        """
        # 0. reset
        self.graph.clear()
        self.steps = []
        self._segmentCache = dict()
        self._resolution = config.get("resolution", 0.05)

        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

        # 2. initial path: straight line from start to goal
        path = [tuple(map(float, checkedStartList[0])), tuple(map(float, checkedGoalList[0]))]

        # 3. bend the path until it is collision-free
        free = False
        for _ in range(config.get("maxIterations", 100)):
            step = self._step(path, config)
            if step is None:
                free = True
                break
            self.steps.append(step)
            path = step["path"]
        else:
            free = self.ratePath(path) >= 0.0

        # 4. store the path as graph (also if it is not collision-free, e.g. for visualization)
        self.path = [list(q) for q in path]
        nodes = ["start"] + list(range(1, len(path) - 1)) + ["goal"]
        for node, q in zip(nodes, self.path):
            self.graph.add_node(node, pos=q)
        self.graph.add_edges_from(zip(nodes[:-1], nodes[1:]))
        return nodes if free else []

    # ------------------------------------------------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------------------------------------------------

    def rateConfiguration(self, q):
        """Q(q) = -penetration depth, collision-free: 0"""
        return -self._collisionChecker.penetrationDepth(list(q))

    def rateSegment(self, qa, qb):
        """Rating of a segment: (Q(q_min), q_min), q_min is its worst configuration (None if it is free).
        The segment is sampled with the given resolution, the results are cached."""
        key = (tuple(qa), tuple(qb))
        if key not in self._segmentCache:
            n = max(1, math.ceil(math.dist(qa, qb) / self._resolution))
            worst = (0.0, None)
            for k in range(n + 1):
                q = tuple(a + k / n * (b - a) for a, b in zip(qa, qb))
                rating = self.rateConfiguration(q)
                if rating < worst[0]:
                    worst = (rating, q)
            self._segmentCache[key] = worst
        return self._segmentCache[key]

    def ratePath(self, path):
        """Rating of a (part of a) path: rating of its worst segment"""
        return min(self.rateSegment(a, b)[0] for a, b in zip(path[:-1], path[1:]))

    # ------------------------------------------------------------------------------------------------------------
    # one step: divide, shift, evaluate, replace
    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def divide(qa, qb, qMin, fDivide):
        """New configurations q_new1 and q_new2 around q_min"""
        qNew1 = tuple(m + fDivide * (a - m) for a, m in zip(qa, qMin))
        qNew2 = tuple(m + fDivide * (b - m) for b, m in zip(qb, qMin))
        return qNew1, qNew2

    @staticmethod
    def shiftDisplacement(qa, qb, config):
        """d = fStep * length of the segment, limited to [deltaMin, deltaMax]"""
        return min(config["deltaMax"], max(config["deltaMin"], config["fStep"] * math.dist(qa, qb)))

    @staticmethod
    def alternatives(qa, qb, d, moveA=True, moveB=True):
        """Shift qa and/or qb perpendicular to the segment by +-d.
        Returns a list of (case, ca, cb): case 1 both points, case 2 only qa, case 3 only qb.
        2D: 4 + 2 + 2 = 8 alternatives (nD: for every perpendicular direction)."""
        result = []
        for n in perpendicularDirections(qa, qb):
            shift = lambda q, s: tuple(float(x) for x in np.asarray(q) + s * d * n)
            signs = (1, -1)
            if moveA and moveB:
                result += [(1, shift(qa, sa), shift(qb, sb)) for sa in signs for sb in signs]
            if moveA:
                result += [(2, shift(qa, s), tuple(qb)) for s in signs]
            if moveB:
                result += [(3, tuple(qa), shift(qb, s)) for s in signs]
        return result

    def _step(self, path, config, eps=1e-9):
        """One step of the BB-Method. Returns None, if the path is collision-free, otherwise a dict:

        previous: the path before the step
        segment: index i of the worst segment (path[i], path[i+1]), qa, qb, qMin, rating: Q(q_min)
        divided: False, if q_min is a point of the path, which is shifted instead of dividing the segment
        dividedPath: path after dividing, ia, ib: indices of the points to shift in dividedPath
        d: shifting displacement, currentRating: rating of the changed part of the path before shifting
        alternatives: list of dict(case, ca, cb, subPath, rating, length), best: the best one of them
        replaced: True, if the best alternative is better than the current rating, path: the new path
        """
        ratings = [self.rateSegment(a, b) for a, b in zip(path[:-1], path[1:])]
        i = min(range(len(ratings)), key=lambda k: ratings[k][0])
        rating, qMin = ratings[i]
        if rating >= 0.0:
            return None
        qa, qb = path[i], path[i + 1]

        atA = math.dist(qMin, qa) < eps and i > 0
        atB = math.dist(qMin, qb) < eps and i + 1 < len(path) - 1
        if atA or atB:
            # q_min is a point of the path: this point is shifted instead of dividing the segment
            dividedPath, ia, ib, divided = list(path), i, i + 1, False
            moveA, moveB = atA, not atA
        else:
            qNew1, qNew2 = self.divide(qa, qb, qMin, config["fDivide"])
            dividedPath, ia, ib, divided = path[:i + 1] + [qNew1, qNew2] + path[i + 1:], i + 1, i + 2, True
            moveA, moveB = True, True

        d = self.shiftDisplacement(dividedPath[ia], dividedPath[ib], config)
        lo, hi = max(ia - 1, 0), min(ib + 1, len(dividedPath) - 1)   # q_prev, ..., q_next
        current = dividedPath[lo:hi + 1]
        currentRating = self.ratePath(current)

        alternatives = []
        for case, ca, cb in self.alternatives(dividedPath[ia], dividedPath[ib], d, moveA, moveB):
            subPath = list(current)
            subPath[ia - lo], subPath[ib - lo] = ca, cb
            alternatives.append(dict(case=case, ca=ca, cb=cb, subPath=subPath, rating=self.ratePath(subPath),
                                     length=pathLength(subPath)))
        # highest rating, among equal ratings the shortest
        best = max(alternatives, key=lambda a: (round(a["rating"], 9), -a["length"]))

        newPath = list(dividedPath)
        replaced = best["rating"] > currentRating
        if replaced:
            newPath[ia], newPath[ib] = best["ca"], best["cb"]
        return dict(previous=list(path), segment=i, qa=qa, qb=qb, qMin=qMin, rating=rating, divided=divided,
                    dividedPath=dividedPath, ia=ia, ib=ib, d=d, currentRating=currentRating,
                    alternatives=alternatives, best=best, replaced=replaced, path=newPath)
