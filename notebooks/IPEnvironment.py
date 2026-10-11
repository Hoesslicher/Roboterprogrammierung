# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPerfMonitor import IPPerfMonitor

from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union
from shapely import plotting

# color of the obstacles in all notebooks (start: green, goal: red)
OBSTACLE_COLOR = "#52514e"

# penetration depth of a configuration on the border of an obstacle: it is in collision, but its distance to the
# free space is 0
MIN_PENETRATION_DEPTH = 1e-9

class CollisionChecker(object):

    def __init__(self, scene, limits=None, statistic=None):
        self.scene = scene
        self.limits = [[0.0, 22.0], [0.0, 22.0]] if limits is None else limits
        self.statistic = statistic
        self._freeSpace = None
        self._freeSpaceKey = None

    def getDim(self):
        """Return dimension of Environment (Shapely should currently always be 2)."""
        return 2

    def getEnvironmentLimits(self):
        """Return limits of Environment."""
        return list(self.limits)

    def isInLimits(self, pos):
        """Return whether a configuration lies inside the configured limits."""
        assert (len(pos) == self.getDim())
        return (self.limits[0][0] <= pos[0] <= self.limits[0][1]
                and self.limits[1][0] <= pos[1] <= self.limits[1][1])

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """Return whether a configuration is invalid.
        Collision or outside limits -> True
        Free and inside limits -> False
        """
        assert (len(pos) == self.getDim())
        if not self.isInLimits(pos):
            return True
        for key, value in self.scene.items():
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos, steps=40):
        """Check a line by sampling intermediate points.
        This illustrates local planner discretization, but it may miss thin obstacles.
        """
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        dim = self.getDim()
        for i in range(steps + 1):
            t = i / steps
            testPoint = [startPos[d] + t * (endPos[d] - startPos[d]) for d in range(dim)]
            if self.pointInCollision(testPoint):
                return True
        return False

    @IPPerfMonitor
    def lineInCollisionExact(self, startPos, endPos):
        """Check whether the exact line segment from startPos to endPos collides."""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        if self.pointInCollision(startPos) or self.pointInCollision(endPos):
            return True
        line = LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])
        for key, value in self.scene.items():
            if value.intersects(line):
                return True
        return False

    def getFreeSpace(self):
        """Return the free space as Shapely geometry: the area inside the limits minus all obstacles.
        Overlapping obstacles are merged into one. The result is cached until the scene or the limits change.
        """
        key = (tuple(map(id, self.scene.values())), tuple(map(tuple, self.limits)))
        if key != self._freeSpaceKey:
            area = box(self.limits[0][0], self.limits[1][0], self.limits[0][1], self.limits[1][1])
            self._freeSpace = area.difference(unary_union(list(self.scene.values())))
            self._freeSpaceKey = key
        return self._freeSpace

    @IPPerfMonitor
    def penetrationDepth(self, pos):
        """Return the (virtual) penetration depth of a configuration, e.g. for the BB-Method.
        Free -> 0.0
        In collision or outside limits -> distance to the free space, i.e. how far the configuration
        has to be moved at least to become free.
        Valid for a point robot (C-space = work space); collision checkers for robots with an extent
        have to override it (see KinChainCollisionChecker).
        """
        assert (len(pos) == self.getDim())
        if not self.pointInCollision(pos):
            return 0.0
        return max(self.getFreeSpace().distance(Point(pos[0], pos[1])), MIN_PENETRATION_DEPTH)

    def drawObstacles(self, ax):
        for key, value in self.scene.items():
            plotting.plot_polygon(value, add_points=False, ax=ax, color=OBSTACLE_COLOR)
            
