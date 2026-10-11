# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Visualization helpers for the notebook IP-9-0-RRT-Basics: the RRT step by step.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely import plotting, voronoi_polygons
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import unary_union

from IPRRT import nearestPointOfTree, splitEdge
from IPVISPRMBasics import drawEnvironment, lectureScene, plotEnvironment, setupAxes
from IPVISStyle import (COLLISION_COLOR, EDGE_COLOR, FREE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR, PATH_COLOR,
                        START_COLOR)

SAMPLE_COLOR = FREE_COLOR


# ---------------------------------------------------------------------------
# RRT with recording of every step
# ---------------------------------------------------------------------------

def traceRRT(collisionChecker, start, goal, numberOfGeneratedNodes, testGoalAfterNumberOfNodes, extendFraction=1.0,
             seed=None, collisionCheckingSteps=40, nearestPoint=False):
    """RRT as implemented in IPRRT.py (extendFraction 1.0: RRTSimple, 0.5: RRT, nearestPoint=True: RRTEdges), but
    every step is recorded.

    Returns (tree, events, path). Node 0 is the start, "goal" the goal (if it was connected), nodes inserted into
    edges are "split0", "split1", ... Events:
    {"type": "extend", "qRand": pos, "near": node, "qNear": pos, "edge": edge, "qNew": pos,
     "result": "added" | "collision", "node": node}
    {"type": "goalTest", "near": node, "qNear": pos, "edge": edge, "result": "connected" | "collision"}
    edge is None, if qNear is a node. Otherwise qNear lies inside the edge (u, v): if the connection is added, the
    edge is split by the node near, if it is in collision, near is None.
    """
    rng = random.Random(seed)
    limits = collisionChecker.getEnvironmentLimits()
    tree = nx.Graph()
    tree.add_node(0, pos=list(start))
    events = []
    generated = 1   # start and added random configurations, the nodes inserted into edges are not counted
    splits = 0

    def nearest(q):
        if nearestPoint:
            return nearestPointOfTree(tree, q)
        positions = [tree.nodes[i]["pos"] for i in range(generated)]
        near = int(cKDTree(positions).query(q, k=1)[1])
        return positions[near], near, None

    def insert(qNear, near, edge):
        nonlocal splits
        if edge is None:
            return near
        name = "split" + str(splits)
        splitEdge(tree, qNear, edge, name)
        splits += 1
        return name

    while generated < numberOfGeneratedNodes:
        if generated % testGoalAfterNumberOfNodes == 0:
            qNear, near, edge = nearest(goal)
            connected = not collisionChecker.lineInCollision(qNear, goal, steps=collisionCheckingSteps)
            if connected:
                near = insert(qNear, near, edge)
            events.append({"type": "goalTest", "near": near, "qNear": list(qNear), "edge": edge,
                           "result": "connected" if connected else "collision"})
            if connected:
                tree.add_node("goal", pos=list(goal))
                tree.add_edge(near, "goal")
                return tree, events, nx.shortest_path(tree, 0, "goal")

        qRand = [rng.uniform(low, high) for low, high in limits]
        while collisionChecker.pointInCollision(qRand):
            qRand = [rng.uniform(low, high) for low, high in limits]
        qNear, near, edge = nearest(qRand)
        if extendFraction == 1.0:
            qNew = qRand
        else:
            qNew = list(extendFraction * (np.array(qRand) - np.array(qNear)) + np.array(qNear))
        added = not collisionChecker.lineInCollision(qNear, qNew, steps=collisionCheckingSteps)
        if added:
            near = insert(qNear, near, edge)
        events.append({"type": "extend", "qRand": qRand, "near": near, "qNear": list(qNear), "edge": edge,
                       "qNew": qNew, "result": "added" if added else "collision",
                       "node": generated if added else None})
        if added:
            tree.add_node(generated, pos=qNew)
            tree.add_edge(near, generated)
            generated += 1
    return tree, events, []


def treeAtStep(start, events, step):
    """Tree after the given event (without the goal)."""
    tree = nx.Graph()
    tree.add_node(0, pos=list(start))
    for event in events[:step + 1]:
        if event["result"] in ("added", "connected") and event.get("edge") is not None:
            splitEdge(tree, event["qNear"], event["edge"], event["near"])
        if event["type"] == "extend" and event["result"] == "added":
            tree.add_node(event["node"], pos=event["qNew"])
            tree.add_edge(event["near"], event["node"])
    return tree


def treeLength(tree):
    """Sum of the lengths of all edges"""
    return sum(math.dist(tree.nodes[u]["pos"], tree.nodes[v]["pos"]) for u, v in tree.edges())


def keySteps(events):
    """Index of the first event of every typical situation (used for the figures)."""
    steps = dict()
    numberOfNodes = 1
    for k, event in enumerate(events):
        if event["type"] == "goalTest":
            steps.setdefault("goal" + event["result"].capitalize(), k)
        elif event["result"] == "added":
            if numberOfNodes == 1:
                steps.setdefault("first", k)
            elif numberOfNodes >= 4:
                steps.setdefault("split" if event.get("edge") is not None else "added", k)
            numberOfNodes += 1
        else:
            steps.setdefault("collision", k)
    steps["last"] = len(events) - 1
    return steps


CAPTIONS = {
    "first": r"tree = {$q_s$}: $q_{near} = q_s$, the extension is added",
    "added": r"new node: $q_{near}$ is the nearest node of the tree to $q_{rand}$",
    "split": r"$q_{near}$ lies inside an edge: the edge is split, $q_{near}$ becomes a new node",
    "collision": r"extension in collision: $q_{new}$ is discarded",
    "goalCollision": r"goal test: connection from the nearest node to $q_g$ in collision",
    "goalConnected": r"goal test successful: path from $q_s$ to $q_g$",
    "last": "final tree",
}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def drawTree(ax, tree, nodeSize=40, edgeWidth=1.2):
    pos = nx.get_node_attributes(tree, "pos")
    nx.draw_networkx_edges(tree, pos, ax=ax, edge_color=EDGE_COLOR, width=edgeWidth)
    nx.draw_networkx_nodes(tree, pos, ax=ax, node_size=nodeSize, node_color=NODE_COLOR, edgecolors=EDGE_COLOR,
                           linewidths=0.8)


def drawStartGoalPoints(ax, start, goal, size=130):
    for q, color, label in ((start, START_COLOR, "$q_s$"), (goal, GOAL_COLOR, "$q_g$")):
        if q is None:
            continue
        ax.scatter(*q, s=size, color=color, edgecolor="white", linewidth=1.0, zorder=8)
        ax.annotate(label, q, xytext=(6, 6), textcoords="offset points", fontsize=12, zorder=8)


def drawPath(ax, tree, path, width=4.0):
    if path:
        pos = nx.get_node_attributes(tree, "pos")
        nx.draw_networkx_edges(tree, pos, ax=ax, edgelist=list(zip(path[:-1], path[1:])), edge_color=PATH_COLOR,
                               width=width)


def drawNearestNode(ax, tree, q):
    """Connection from q to the nearest node of the tree, for comparison with the nearest point"""
    pos = nx.get_node_attributes(tree, "pos")
    node = min(pos, key=lambda n: math.dist(pos[n], q))
    ax.plot([q[0], pos[node][0]], [q[1], pos[node][1]], color=MUTED, linestyle="--", linewidth=1.5, zorder=5)
    ax.scatter(*pos[node], s=120, facecolor="none", edgecolor=MUTED, linewidth=1.5, zorder=7)
    ax.annotate("nearest node", pos[node], xytext=(8, 8), textcoords="offset points", ha="left", fontsize=10,
                color=MUTED, zorder=7)


def drawRRTState(ax, collisionChecker, start, goal, events, step, caption=None, highlight=True, nearestNode=False):
    """State of the RRT after the given event (event `step` is highlighted).
    nearestNode: show also the connection to the nearest node (for the variant with the nearest point)."""
    final = step == len(events) - 1 and events[step]["type"] == "goalTest" and events[step]["result"] == "connected"
    tree = treeAtStep(start, events, step)
    pos = nx.get_node_attributes(tree, "pos")
    event = events[step]

    drawEnvironment(ax, collisionChecker)
    drawTree(ax, tree)
    if final:
        tree.add_node("goal", pos=list(goal))
        tree.add_edge(event["near"], "goal")
        drawPath(ax, tree, nx.shortest_path(tree, 0, "goal"))
    drawStartGoalPoints(ax, start, goal)

    if highlight:
        qNear = event["qNear"]
        if nearestNode:
            drawNearestNode(ax, treeAtStep(start, events, step - 1),
                            event["qRand"] if event["type"] == "extend" else goal)
        ax.scatter(*qNear, s=120, facecolor="none", edgecolor=SAMPLE_COLOR, linewidth=1.8, zorder=7)
        ax.annotate("$q_{near}$", qNear, xytext=(-8, -16), textcoords="offset points", fontsize=11,
                    color=SAMPLE_COLOR, zorder=7)
        if event["type"] == "extend":
            qRand, qNew = event["qRand"], event["qNew"]
            if math.dist(qRand, qNew) > 1e-9:
                ax.plot([qNew[0], qRand[0]], [qNew[1], qRand[1]], color=MUTED, linestyle=":", linewidth=1.5, zorder=5)
                ax.annotate("$q_{new}$", qNew, xytext=(-14, 9), textcoords="offset points", fontsize=11,
                            color=SAMPLE_COLOR, zorder=7)
            ax.scatter(*qRand, marker="x", s=70, linewidths=2, color=SAMPLE_COLOR, zorder=9)
            ax.annotate("$q_{rand}$", qRand, xytext=(6, 6), textcoords="offset points", fontsize=11,
                        color=SAMPLE_COLOR, zorder=7)
            added = event["result"] == "added"
            ax.plot([qNear[0], qNew[0]], [qNear[1], qNew[1]], zorder=6,
                    color=SAMPLE_COLOR if added else COLLISION_COLOR, linestyle="-" if added else "--",
                    linewidth=2.5 if added else 2.0)
            ax.scatter(*qNew, s=35, color=SAMPLE_COLOR if added else COLLISION_COLOR, edgecolor="white", zorder=7)
        elif not final:
            ax.plot([qNear[0], goal[0]], [qNear[1], goal[1]], color=COLLISION_COLOR, linestyle="--", linewidth=2.0,
                    zorder=6)

    if caption is None:
        caption = f"step {step}: " + (f"extension {event['result']}" if event["type"] == "extend"
                                     else f"goal test: {event['result']}")
    ax.set_title(caption, fontsize=10, loc="left")


def legendRRT(nearestPoint=False):
    return [Line2D([], [], marker="x", linestyle="", color=SAMPLE_COLOR, markersize=8, markeredgewidth=2,
                   label="$q_{rand}$: random free configuration"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=SAMPLE_COLOR,
                   markersize=10, markeredgewidth=1.8,
                   label="$q_{near}$: nearest " + ("point" if nearestPoint else "node") + " of the tree"),
            Line2D([], [], color=SAMPLE_COLOR, linewidth=2.5, label="extension added"),
            Line2D([], [], color=COLLISION_COLOR, linestyle="--", linewidth=2.0, label="connection in collision"),
            Line2D([], [], color=MUTED, linestyle=":", linewidth=1.5, label="not extended (only part of the way)")]


def plotStep(collisionChecker, start, goal, events, step, caption=None, legend=True, highlight=True,
             figsize=(9.5, 4.2), nearestNode=False):
    """One step of the RRT (also used for the interactive slider)."""
    fig, ax = plt.subplots(figsize=figsize)
    drawRRTState(ax, collisionChecker, start, goal, events, step, caption=caption, highlight=highlight,
                 nearestNode=nearestNode)
    if legend:
        handles = legendRRT(any(event.get("edge") is not None for event in events))
        if nearestNode:
            handles.append(Line2D([], [], color=MUTED, linestyle="--", linewidth=1.5,
                                  label="connection to the nearest node"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def plotSituation(collisionChecker, start, goal, events, situation, legend=False):
    """The first step of a typical situation, see keySteps(). For "split" the nearest node is shown as well."""
    plotStep(collisionChecker, start, goal, events, keySteps(events)[situation], caption=CAPTIONS[situation],
             legend=legend, highlight=situation != "last", nearestNode=situation == "split")


# ---------------------------------------------------------------------------
# Voronoi bias
# ---------------------------------------------------------------------------

def voronoiCells(collisionChecker, tree):
    """Voronoi cell of every node of the tree, clipped to the environment: {node: polygon}."""
    limits = collisionChecker.getEnvironmentLimits()
    area = box(limits[0][0], limits[1][0], limits[0][1], limits[1][1])
    pos = nx.get_node_attributes(tree, "pos")
    nodes = list(pos)
    cells = voronoi_polygons(MultiPoint([pos[n] for n in nodes]), extend_to=area)
    result = dict()
    for cell in cells.geoms:
        for n in nodes:
            if cell.contains(Point(pos[n])):
                result[n] = cell.intersection(area)
                break
    return result


def showVoronoiBias(collisionChecker, start, tree, sizes=(5, 20, 80), panelWidth=5.2):
    """Tree after a growing number of nodes with the Voronoi cells of its nodes. The darker a cell, the larger it is,
    i.e. the higher the probability, that the node is the nearest one to a random sample."""
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(1, len(sizes), figsize=(len(sizes) * panelWidth, panelWidth * aspect + 0.7))
    obstacles = unary_union(list(collisionChecker.scene.values()))
    freeArea = box(limits[0][0], limits[1][0], limits[0][1], limits[1][1]).difference(obstacles).area
    for ax, size in zip(axes, sizes):
        subtree = tree.subgraph([n for n in tree if isinstance(n, int) and n < size]).copy()
        cells = voronoiCells(collisionChecker, subtree)
        largest = max(cell.area for cell in cells.values())
        for node, cell in cells.items():
            plotting.plot_polygon(cell, ax=ax, add_points=False,
                                  facecolor=to_rgba(SAMPLE_COLOR, 0.08 + 0.5 * cell.area / largest),
                                  edgecolor=to_rgba(SAMPLE_COLOR, 0.6), linewidth=0.6)
        drawEnvironment(ax, collisionChecker, f"{size} nodes")
        drawTree(ax, subtree, nodeSize=25)
        ax.scatter(*start, s=110, color=START_COLOR, edgecolor="white", zorder=8)
        node = max(cells, key=lambda n: cells[n].area)
        probability = cells[node].difference(obstacles).area / freeArea   # samples are only generated in free space
        ax.annotate(f"{probability:.0%}", subtree.nodes[node]["pos"], xytext=(6, 6),
                    textcoords="offset points", fontsize=10, color=EDGE_COLOR)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def pathLength(tree, path):
    pos = nx.get_node_attributes(tree, "pos")
    return sum(math.dist(pos[a], pos[b]) for a, b in zip(path[:-1], path[1:])) if path else float("nan")


def showVariants(collisionChecker, start, goal, results, panelWidth=6.0):
    """Final trees of several runs side by side: results = [(title, tree, path), ...]."""
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * panelWidth, panelWidth * aspect + 0.6))
    for ax, (title, tree, path) in zip(axes, results):
        drawEnvironment(ax, collisionChecker, title)
        drawTree(ax, tree, nodeSize=25)
        drawPath(ax, tree, path)
        drawStartGoalPoints(ax, start, goal)
    fig.tight_layout()
    plt.show()
