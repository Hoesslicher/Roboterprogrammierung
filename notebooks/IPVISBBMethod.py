# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Visualization of the BB-Method (IPBBMethod.py) and helpers for the notebook IP-11-0-BB-Method.
A step of the method is given as dict (see BBMethod._step), the figures work with the steps of BBMethod as well as
with the steps of the implementation in the notebook.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from shapely import plotting
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from IPEnvironment import OBSTACLE_COLOR
from IPVISPRMBasics import drawEnvironment, setupAxes
from IPVISStyle import (COLLISION_COLOR, EDGE_COLOR, FREE_COLOR, GOAL_COLOR, MUTED, NODE_COLOR, PATH_COLOR,
                        SPECIAL_COLOR, START_COLOR, drawStartGoal)

NEW_COLOR = SPECIAL_COLOR      # new configurations q_new1, q_new2 (dividing) and shifted configurations
QMIN_COLOR = COLLISION_COLOR   # worst configuration q_min
# penetration depth: free white, the deeper the darker (one hue)
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth", ["#ffffff", "#f8d5c4", COLLISION_COLOR, "#6b260c"])


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def lectureExample():
    """Obstacle, start and goal of the example of the lecture slides. Returns (scene, limits, start, goal)."""
    raw = [(0.95, 99.7), (29.2, 128.0), (91.5, 102.6), (156.5, 119.5), (213.1, 63.0), (198.9, 29.1),
           (181.2, 29.1), (152.9, 0.8), (63.2, 23.4), (43.4, 71.5)]   # taken from the original slide, y downwards

    def transform(p):
        return (p[0] / 25 + 2.0, (140 - p[1]) / 25 + 2.0)

    scene = {"obstacle": Polygon([transform(p) for p in raw])}
    return scene, [[0.0, 14.0], [0.0, 10.0]], list(transform((-15, -23))), list(transform((265, 159)))


# ---------------------------------------------------------------------------
# basic drawing
# ---------------------------------------------------------------------------

def drawPath(ax, path, color=EDGE_COLOR, width=2.0, nodeSize=30, linestyle="-", nodes=True, zorder=4):
    """Path given as list of configurations"""
    xs, ys = [q[0] for q in path], [q[1] for q in path]
    ax.plot(xs, ys, color=color, linewidth=width, linestyle=linestyle, solid_capstyle="round", zorder=zorder)
    if nodes:
        ax.scatter(xs, ys, s=nodeSize, color=NODE_COLOR, edgecolors=EDGE_COLOR, linewidths=0.8, zorder=zorder + 0.5)


def drawStartGoalPoints(ax, start, goal, size=110, labels=True):
    for q, color, label in ((start, START_COLOR, "$q_{start}$"), (goal, GOAL_COLOR, "$q_{goal}$")):
        ax.scatter(q[0], q[1], s=size, color=color, edgecolor="white", linewidth=1.0, zorder=8)
        if labels:
            ax.annotate(label, (q[0], q[1]), xytext=(6, 6), textcoords="offset points", fontsize=11, zorder=8)


def drawPoint(ax, q, label=None, color=QMIN_COLOR, size=60, offset=(6, -15), zorder=9):
    ax.scatter(q[0], q[1], s=size, color=color, edgecolor=EDGE_COLOR, linewidth=1.0, zorder=zorder)
    if label:
        ax.annotate(label, (q[0], q[1]), xytext=offset, textcoords="offset points", fontsize=11, color=color,
                    zorder=zorder)


def sampleSegment(qa, qb, resolution):
    """Configurations on the segment with a maximal distance of resolution (as in BBMethod.rateSegment)"""
    n = max(1, math.ceil(math.dist(qa, qb) / resolution))
    return [[a + k / n * (b - a) for a, b in zip(qa, qb)] for k in range(n + 1)]


def drawSegmentDepth(ax, collisionChecker, qa, qb, resolution, vmax=None, size=14):
    """The rated configurations of a segment, colored by their penetration depth (free ones are not shown)"""
    qs = sampleSegment(qa, qb, resolution)
    depths = np.array([collisionChecker.penetrationDepth(q) for q in qs])
    inside = depths > 0
    if inside.any():
        ax.scatter([q[0] for q, i in zip(qs, inside) if i], [q[1] for q, i in zip(qs, inside) if i], s=size,
                   c=depths[inside], cmap=DEPTH_CMAP, vmin=-0.15 * (vmax or depths.max()), vmax=vmax or depths.max(),
                   edgecolors="none", zorder=6)


def drawObstacleOutlines(ax, collisionChecker):
    for obstacle in collisionChecker.scene.values():
        plotting.plot_polygon(obstacle, ax=ax, add_points=False, facecolor="none", edgecolor=OBSTACLE_COLOR,
                              linewidth=1.2, zorder=3)


# ---------------------------------------------------------------------------
# blow up: penetration depth of a point robot
# ---------------------------------------------------------------------------

def depthGrid(collisionChecker, resolution=0.1, depthFunction=None):
    """Penetration depth on a grid over the environment (only for illustration, the BB-Method never does this)"""
    depthFunction = collisionChecker.penetrationDepth if depthFunction is None else depthFunction
    limits = collisionChecker.getEnvironmentLimits()
    xs = np.arange(limits[0][0] + resolution / 2, limits[0][1], resolution)
    ys = np.arange(limits[1][0] + resolution / 2, limits[1][1], resolution)
    return np.array([[depthFunction([x, y]) for x in xs] for y in ys])


def drawDepthLandscape(ax, collisionChecker, resolution=0.1, depthFunction=None, vmax=None, colorbar=True,
                       title=None):
    """Penetration depth of every configuration: free white, the deeper the darker"""
    depth = depthGrid(collisionChecker, resolution, depthFunction)
    limits = collisionChecker.getEnvironmentLimits()
    image = ax.imshow(np.ma.masked_equal(depth, 0.0), origin="lower", cmap=DEPTH_CMAP, vmin=0.0,
                      vmax=vmax or depth.max(), extent=[limits[0][0], limits[0][1], limits[1][0], limits[1][1]],
                      interpolation="nearest", zorder=1)
    drawObstacleOutlines(ax, collisionChecker)
    setupAxes(ax, collisionChecker, title)
    if colorbar:
        plt.colorbar(image, ax=ax, shrink=0.8, label="penetration depth")
    return image


def drawPenetrationDepth(ax, collisionChecker, q):
    """Configuration q with its penetration depth and the nearest free configuration (point robot)"""
    depth = collisionChecker.penetrationDepth(q)
    if depth > 0.0:
        nearest = nearest_points(collisionChecker.getFreeSpace(), Point(q[0], q[1]))[0]
        ax.annotate("", xy=(nearest.x, nearest.y), xytext=(q[0], q[1]), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color=EDGE_COLOR, linewidth=1.5, shrinkA=4, shrinkB=0))
    drawPoint(ax, q, color=QMIN_COLOR if depth > 0.0 else FREE_COLOR, size=50)
    ax.annotate(f"{depth:.2f}", (q[0], q[1]), xytext=(6, 6), textcoords="offset points", fontsize=11, zorder=9)


def showPenetrationDepth(collisionChecker, configurations, resolution=0.1, figsize=(8.5, 5.2)):
    """Penetration depth of all configurations (landscape) and of some test configurations (arrow: shortest way
    into the free space)"""
    fig, ax = plt.subplots(figsize=figsize)
    drawDepthLandscape(ax, collisionChecker, resolution, title="penetration depth: distance to the free space")
    for q in configurations:
        drawPenetrationDepth(ax, collisionChecker, q)
    fig.tight_layout()
    plt.show()


def showDepthComparison(collisionChecker, variants, configuration=None, resolution=0.05, panelWidth=6.0):
    """Landscapes of several depth functions side by side: variants = [(title, depthFunction), ...]"""
    depths = [depthGrid(collisionChecker, resolution, f) for _, f in variants]
    vmax = max(d.max() for d in depths)
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(1, len(variants), figsize=(len(variants) * panelWidth, panelWidth * aspect + 0.5))
    for ax, (title, f) in zip(axes, variants):
        drawDepthLandscape(ax, collisionChecker, resolution, f, vmax=vmax, colorbar=False, title=title)
        if configuration is not None:
            drawPoint(ax, configuration, size=50)
            ax.annotate(f"{f(configuration):.2f}", configuration, xytext=(6, 6), textcoords="offset points",
                        fontsize=11, zorder=9)
    fig.tight_layout()
    plt.show()


def showSegmentRatings(collisionChecker, segments, resolution, figsize=(8.5, 5.2)):
    """Segments with their rated configurations and their worst configuration q_min.
    segments = [(qa, qb, rating, qMin), ...]"""
    fig, ax = plt.subplots(figsize=figsize)
    drawEnvironment(ax, collisionChecker)
    worst = min(range(len(segments)), key=lambda k: segments[k][2])
    for k, (qa, qb, rating, qMin) in enumerate(segments):
        drawPath(ax, [qa, qb], color=COLLISION_COLOR if k == worst else EDGE_COLOR, width=3 if k == worst else 2)
        drawSegmentDepth(ax, collisionChecker, qa, qb, resolution)
        if qMin is not None:
            drawPoint(ax, qMin, "$q_{min}$")
        ax.annotate(str(k + 1), (qa[0], qa[1]), xytext=(-14, -4), textcoords="offset points", fontsize=11, zorder=9)
    ax.text(0.02, 0.97, "\n".join(f"segment {k + 1}: Q = {rating:.2f}" + ("  (worst)" if k == worst else "")
                                  for k, (qa, qb, rating, qMin) in enumerate(segments)),
            transform=ax.transAxes, ha="left", va="top", fontsize=10)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# bend: the steps of the method
# ---------------------------------------------------------------------------

PHASES = ("worst", "divide", "shift", "evaluate", "result")


def caption(step, phase, config):
    """Short description of a phase of a step"""
    if phase == "worst":
        return f"worst segment, deepest configuration $q_{{min}}$: Q = {step['rating']:.2f}"
    if phase == "divide":
        if step["divided"]:
            return f"divide at $q_{{min}}$ ($f_{{divide}}$ = {config['fDivide']}): $q_{{new1}}$, $q_{{new2}}$"
        return "$q_{min}$ is a point of the path: shift this point instead of dividing"
    if phase == "shift":
        return f"shift by $\\pm d$, d = {step['d']:.2f}: {len(step['alternatives'])} alternatives"
    if phase == "evaluate":
        free = sum(1 for a in step["alternatives"] if a["rating"] >= 0.0)
        return f"{free} of {len(step['alternatives'])} alternatives free, best: Q = {step['best']['rating']:.2f}"
    if step["replaced"]:
        return f"better than before (Q = {step['currentRating']:.2f}): replace"
    return f"no improvement (Q = {step['currentRating']:.2f}): keep only the division"


def drawStep(ax, collisionChecker, step, phase, config):
    """One phase of a step: "worst", "divide", "shift", "evaluate" or "result" """
    path, ia, ib = step["dividedPath"], step["ia"], step["ib"]
    if phase in ("worst", "divide"):
        drawPath(ax, step["previous"] if phase == "worst" else path)
        drawPath(ax, [step["qa"], step["qb"]], color=COLLISION_COLOR, width=3.5, nodes=False, zorder=4.2)
        drawSegmentDepth(ax, collisionChecker, step["qa"], step["qb"], config["resolution"])
        if phase == "divide" and step["divided"]:
            drawPath(ax, path[ia - 1:ib + 2], nodes=False, zorder=4.4)
            drawPoint(ax, path[ia], "$q_{new1}$", color=NEW_COLOR, offset=(-12, 9))
            drawPoint(ax, path[ib], "$q_{new2}$", color=NEW_COLOR, offset=(-4, 9))
        drawPoint(ax, step["qMin"], "$q_{min}$")
    elif phase in ("shift", "evaluate"):
        drawPath(ax, path, color=MUTED, width=1.5, nodes=False)
        for alternative in step["alternatives"]:
            if phase == "shift":
                drawPath(ax, alternative["subPath"], color=MUTED, width=1.2, linestyle="--", nodes=False, zorder=3)
            elif alternative is not step["best"]:
                free = alternative["rating"] >= 0.0
                drawPath(ax, alternative["subPath"], color=MUTED if free else COLLISION_COLOR, width=1.2,
                         linestyle="-" if free else "--", nodes=False, zorder=3)
            for q in (alternative["ca"], alternative["cb"]):
                ax.scatter(q[0], q[1], s=18, color=EDGE_COLOR, zorder=5)
        if phase == "shift":
            # arrows from the shifted configurations to their alternatives
            for q, key in ((path[ia], "ca"), (path[ib], "cb")):
                targets = {alternative[key] for alternative in step["alternatives"] if alternative[key] != q}
                for c in targets:
                    ax.annotate("", xy=(c[0], c[1]), xytext=(q[0], q[1]), zorder=4, arrowprops=dict(
                        arrowstyle="-|>", color=MUTED, linewidth=1.0, shrinkA=3, shrinkB=3))
                if targets:
                    drawPoint(ax, q, color=NEW_COLOR, size=45)
        else:
            drawPath(ax, step["best"]["subPath"], color=PATH_COLOR, width=3.0, nodes=False, zorder=4.5)
    else:
        drawPath(ax, step["path"])
        if step["replaced"]:
            drawPath(ax, step["best"]["subPath"], color=PATH_COLOR, width=3.0, zorder=4.6)


def stepFrames(steps):
    """All frames of a run for the interactive slider: ("start", None), (phase, step index), ..., ("end", None)"""
    return [("start", None)] + [(phase, k) for k in range(len(steps)) for phase in PHASES] + [("end", None)]


def legendSteps():
    return [Line2D([], [], color=COLLISION_COLOR, linewidth=3.5, label="worst segment"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=QMIN_COLOR, markeredgecolor=EDGE_COLOR,
                   markersize=8, label="$q_{min}$: deepest configuration"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=NEW_COLOR, markeredgecolor=EDGE_COLOR,
                   markersize=8, label="new / shifted configurations"),
            Line2D([], [], color=MUTED, linestyle="--", linewidth=1.2, label="alternative"),
            Line2D([], [], color=COLLISION_COLOR, linestyle="--", linewidth=1.2, label="alternative in collision"),
            Line2D([], [], color=PATH_COLOR, linewidth=3.0, label="best alternative")]


def _zoom(ax, step, margin=1.0):
    """Limit the axes to the changed part of the path"""
    points = step["dividedPath"][max(step["ia"] - 1, 0):step["ib"] + 2]
    points += [q for a in step["alternatives"] for q in (a["ca"], a["cb"])]
    xs, ys = [q[0] for q in points], [q[1] for q in points]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def drawFrame(ax, collisionChecker, steps, finalPath, frame, config, zoom=False):
    phase, k = frame
    drawEnvironment(ax, collisionChecker)
    start, goal = finalPath[0], finalPath[-1]
    if phase == "start":
        drawPath(ax, [start, goal])
        title = "start: straight line from $q_{start}$ to $q_{goal}$"
    elif phase == "end":
        drawPath(ax, finalPath, color=PATH_COLOR, width=3.0)
        title = f"end after {len(steps)} steps: {len(finalPath)} configurations on the path"
    else:
        drawStep(ax, collisionChecker, steps[k], phase, config)
        title = f"step {k + 1}: " + caption(steps[k], phase, config)
        if zoom:
            _zoom(ax, steps[k])
    drawStartGoalPoints(ax, start, goal)
    ax.set_title(title, fontsize=10, loc="left")


def plotFrame(collisionChecker, steps, finalPath, frame, config, legend=True, figsize=(10, 5.2)):
    """One frame of a run (also used for the interactive slider), see stepFrames()"""
    fig, ax = plt.subplots(figsize=figsize)
    drawFrame(ax, collisionChecker, steps, finalPath, frame, config)
    if legend:
        ax.legend(handles=legendSteps(), loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


def showStep(collisionChecker, step, config, phases=("divide", "shift", "evaluate", "result"), zoom=True,
             columns=2, panelWidth=6.0):
    """The phases of one step side by side"""
    rows = math.ceil(len(phases) / columns)
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth * aspect + 0.5)),
                             squeeze=False)
    for ax, phase in zip(axes.flat, phases):
        drawEnvironment(ax, collisionChecker)
        drawStep(ax, collisionChecker, step, phase, config)
        if zoom:
            _zoom(ax, step)
        ax.set_title(caption(step, phase, config), fontsize=10, loc="left")
    for ax in list(axes.flat)[len(phases):]:
        ax.axis("off")
    fig.legend(handles=legendSteps(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def showSteps(collisionChecker, steps, finalPath, columns=4, panelWidth=4.0):
    """Small multiples: the path after every step (blue: replaced part)"""
    panels = len(steps) + 1
    rows = math.ceil(panels / columns)
    limits = collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth * aspect + 0.45)),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= panels:
            ax.axis("off")
            continue
        drawEnvironment(ax, collisionChecker)
        if k == 0:
            drawPath(ax, [finalPath[0], finalPath[-1]], nodeSize=15)
            ax.set_title("start", fontsize=10, loc="left")
        else:
            step = steps[k - 1]
            drawStep(ax, collisionChecker, step, "result", config={})
            drawPoint(ax, step["qMin"], size=35)
            ax.set_title(f"step {k}: " + ("replace" if step["replaced"] else "only divide"), fontsize=10, loc="left")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def bbVisualize(planner, solution, ax=None, nodeSize=40):
    """Path of the BB-Method (same interface as the visualizations of the other planners).
    If no solution was found, the segments in collision are drawn dashed."""
    graph = planner.graph
    planner._collisionChecker.drawObstacles(ax)
    pos = nx.get_node_attributes(graph, "pos")
    path = planner.path
    for a, b in zip(path[:-1], path[1:]):
        free = planner.rateSegment(a, b)[0] >= 0.0
        ax.plot([a[0], b[0]], [a[1], b[1]], zorder=4, linewidth=3.0 if free else 2.0, linestyle="-" if free else "--",
                color=(PATH_COLOR if solution else EDGE_COLOR) if free else COLLISION_COLOR)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=nodeSize, node_color=NODE_COLOR, edgecolors=EDGE_COLOR,
                           linewidths=0.8)
    drawStartGoal(graph, pos, ax, nodeSize)


def showResults(results, columns=None, panelWidth=4.5):
    """Results of several runs side by side: results = [(title, planner, solution), ...]"""
    columns = columns or len(results)
    rows = math.ceil(len(results) / columns)
    limits = results[0][1]._collisionChecker.getEnvironmentLimits()
    aspect = (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0])
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth * aspect + 0.6)),
                             squeeze=False)
    for ax, (title, planner, solution) in zip(axes.flat, results):
        bbVisualize(planner, solution, ax=ax, nodeSize=25)
        setupAxes(ax, planner._collisionChecker, title)
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def showLocalMinimum(planner, config, margin=1.5, panelWidth=6.0):
    """Left: the path, on which the method got stuck, right: its last step (evaluation of the alternatives)"""
    collisionChecker = planner._collisionChecker
    step = planner.steps[-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth, panelWidth + 0.4))
    bbVisualize(planner, [], ax=ax1, nodeSize=25)
    drawPoint(ax1, step["qMin"], "$q_{min}$", size=45)
    setupAxes(ax1, collisionChecker, f"path after {len(planner.steps)} steps")
    drawEnvironment(ax2, collisionChecker)
    drawStep(ax2, collisionChecker, step, "evaluate", config)
    drawPoint(ax2, step["qMin"], "$q_{min}$", size=45)
    _zoom(ax2, step, margin)
    ax2.set_title("last step (zoomed): " + caption(step, "evaluate", config) + "\n"
                  + caption(step, "result", config), fontsize=10, loc="left")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# planar robot: blow up
# ---------------------------------------------------------------------------

def drawPlanarRobot(ax, collisionChecker, pos, color=NODE_COLOR, alpha=1.0, blowUp=False, zorder=5):
    """Planar robot of a KinChainCollisionChecker with capsule links at configuration pos.
    blowUp: show the result of blowUp(): the first colliding link only blown up as far as possible (filled),
    its full size dashed"""
    chain = collisionChecker.kin_chain
    chain.move(pos)
    joints = [np.asarray(p, dtype=float) for p in chain.get_transforms()]
    radius = max(collisionChecker.link_radius, 0.02)
    iColl, scale = collisionChecker.blowUp(pos) if blowUp else (None, 1.0)
    for i in range(1, len(joints)):
        a, b = joints[i - 1], joints[i]
        link = LineString([tuple(a), tuple(b)])
        if iColl is None or i < iColl:
            plotting.plot_polygon(link.buffer(radius), ax=ax, add_points=False, facecolor=color, alpha=alpha,
                                  edgecolor=EDGE_COLOR, linewidth=1.0, zorder=zorder)
        else:
            plotting.plot_polygon(link.buffer(radius), ax=ax, add_points=False, facecolor="none", linestyle="--",
                                  edgecolor=COLLISION_COLOR if i == iColl else MUTED, linewidth=1.2, zorder=zorder)
            if i == iColl and scale > 0.0:
                if collisionChecker.skeleton == "joint":
                    blown = LineString([tuple(a), tuple(a + scale * (b - a))]).buffer(scale * radius)
                else:
                    blown = link.buffer(scale * radius)
                plotting.plot_polygon(blown, ax=ax, add_points=False, facecolor=COLLISION_COLOR, alpha=0.6,
                                      edgecolor=COLLISION_COLOR, linewidth=1.0, zorder=zorder + 0.1)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE_COLOR, linewidth=1.0, alpha=alpha, zorder=zorder + 0.2)
    ax.scatter([p[0] for p in joints[:-1]], [p[1] for p in joints[:-1]], s=25, color="white",
               edgecolors=EDGE_COLOR, linewidths=1.0, alpha=alpha, zorder=zorder + 0.3)


def workspaceLimits(collisionChecker, configurations, margin=0.3):
    """Square limits of the work space containing the robot at all configurations and all obstacles"""
    chain = collisionChecker.kin_chain
    points = []
    for pos in configurations:
        chain.move(pos)
        points += [tuple(p) for p in chain.get_transforms()]
    for obstacle in collisionChecker.scene.values():
        bounds = obstacle.bounds
        points += [(bounds[0], bounds[1]), (bounds[2], bounds[3])]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + collisionChecker.link_radius + margin
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [[cx - half, cx + half], [cy - half, cy + half]]


def setupWorkspace(ax, collisionChecker, title=None, limits=None):
    """Obstacles in the work space, default limits: reach of the robot"""
    if limits is None:
        reach = sum(joint.a for joint in collisionChecker.kin_chain.joints) + collisionChecker.link_radius + 0.2
        limits = [[-reach, reach], [-reach, reach]]
    collisionChecker.drawObstacles(ax, inWorkspace=True)
    ax.scatter(0, 0, s=120, marker="s", color=EDGE_COLOR, zorder=4)   # base of the robot
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def blowUpText(collisionChecker, pos):
    iColl, scale = collisionChecker.blowUp(pos)
    if iColl is None:
        return f"free: $Q_{{scale}}$ = {collisionChecker.getDim()} = dim(C)"
    return f"$i_{{coll}}$ = {iColl}, scale = {scale:.2f}: $Q_{{scale}}$ = {iColl - 1 + scale:.2f}"


def showBlowUp(collisionChecker, configurations, panelWidth=4.2):
    """Blow up of the robot at some configurations"""
    fig, axes = plt.subplots(1, len(configurations), figsize=(len(configurations) * panelWidth, panelWidth + 0.4))
    limits = workspaceLimits(collisionChecker, configurations)
    for ax, pos in zip(np.atleast_1d(axes), configurations):
        setupWorkspace(ax, collisionChecker, blowUpText(collisionChecker, pos), limits)
        drawPlanarRobot(ax, collisionChecker, pos, blowUp=True)
    fig.tight_layout()
    plt.show()


def drawQScaleLandscape(ax, collisionChecker, n=120, colorbar=True, title=None, vmin=None):
    """Q_scale of every configuration of a robot with two joints (only for illustration).
    Colors from vmin (default: Q_scale of the lowest colored link, rounded down) to dim(C) (white: free)"""
    limits = collisionChecker.getEnvironmentLimits()
    q1 = np.linspace(limits[0][0], limits[0][1], n)
    q2 = np.linspace(limits[1][0], limits[1][1], n)
    qScale = np.array([[collisionChecker.qScale([a, b]) for a in q1] for b in q2])
    vmin = min(math.floor(qScale.min()), collisionChecker.getDim() - 1) if vmin is None else vmin
    image = ax.imshow(qScale, origin="lower", cmap=DEPTH_CMAP.reversed(), vmin=vmin, vmax=collisionChecker.getDim(),
                      extent=[limits[0][0], limits[0][1], limits[1][0], limits[1][1]], interpolation="nearest",
                      zorder=1)
    setupAxes(ax, collisionChecker, title)
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    if colorbar:
        plt.colorbar(image, ax=ax, shrink=0.8, label="$Q_{scale}$")


def interpolatePath(path, count):
    """count configurations with equal distances along the path"""
    lengths = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(path[:-1], path[1:])])
    result = []
    for s in np.linspace(0.0, lengths[-1], count):
        k = min(int(np.searchsorted(lengths, s, side="right")) - 1, len(path) - 2)
        t = (s - lengths[k]) / (lengths[k + 1] - lengths[k]) if lengths[k + 1] > lengths[k] else 0.0
        result.append([a + t * (b - a) for a, b in zip(path[k], path[k + 1])])
    return result


def poseColor(collisionChecker, configurations, k):
    """start green, goal red, in collision orange, otherwise the node color"""
    if collisionChecker.pointInCollision(configurations[k]):
        return COLLISION_COLOR
    return START_COLOR if k == 0 else GOAL_COLOR if k == len(configurations) - 1 else NODE_COLOR


def drawRobotMotion(ax, collisionChecker, path, poses=9, title=None, limits=None):
    """The robot at configurations with equal distances along the path (the later, the more opaque)"""
    configurations = interpolatePath(path, poses)
    setupWorkspace(ax, collisionChecker, title, limits or workspaceLimits(collisionChecker, configurations))
    for k, pos in enumerate(configurations):
        drawPlanarRobot(ax, collisionChecker, pos, color=poseColor(collisionChecker, configurations, k),
                        alpha=0.35 + 0.65 * k / (len(configurations) - 1), zorder=5 + k)


def showRobotPath(collisionChecker, path, title=None, poses=9, n=120, panelWidth=5.2):
    """Left: path in the C-space (joint angles) on the Q_scale landscape, right: the robot along the path"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * panelWidth + 1.0, panelWidth))
    drawQScaleLandscape(ax1, collisionChecker, n, title=title or "C-space")
    drawPath(ax1, path, nodeSize=20)
    drawStartGoalPoints(ax1, path[0], path[-1])
    drawRobotMotion(ax2, collisionChecker, path, poses, "work space: robot along the path (orange: in collision)")
    fig.tight_layout()
    plt.show()


def showRobotMotions(collisionChecker, motions, poses=12, panelWidth=5.0):
    """Robot motions side by side (same limits): motions = [(title, path), ...]"""
    limits = workspaceLimits(collisionChecker, [q for _, path in motions for q in interpolatePath(path, poses)])
    fig, axes = plt.subplots(1, len(motions), figsize=(len(motions) * panelWidth, panelWidth + 0.3))
    for ax, (title, path) in zip(np.atleast_1d(axes), motions):
        drawRobotMotion(ax, collisionChecker, path, poses, title, limits)
    fig.tight_layout()
    plt.show()


def showRobotFilmStrip(collisionChecker, path, frames=8, columns=4, panelWidth=3.2):
    """The motion as small multiples: one configuration per panel (the previous one faint)"""
    configurations = interpolatePath(path, frames)
    limits = workspaceLimits(collisionChecker, configurations)
    rows = math.ceil(frames / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * (panelWidth + 0.3)),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= frames:
            ax.axis("off")
            continue
        setupWorkspace(ax, collisionChecker, f"{k + 1} / {frames}", limits)
        if k > 0:
            drawPlanarRobot(ax, collisionChecker, configurations[k - 1], color=MUTED, alpha=0.25)
        drawPlanarRobot(ax, collisionChecker, configurations[k], color=poseColor(collisionChecker, configurations, k),
                        zorder=6)
    fig.tight_layout()
    plt.show()
