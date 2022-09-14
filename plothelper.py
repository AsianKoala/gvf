from math import radians
import numpy as np
from path import *
from geometry import * 
from koawalib import *


def plotPath(path):
    len = path.length()
    xPath, yPath = [], []
    displacements = np.linspace(0.0, len, num=int(len*2))
    for s in displacements:
        v = path.get(float(s))
        xPath.append(v.x)
        yPath.append(v.y)
    plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.3)

def plotRobot(startPose, controller: koawalib_gvf, path: Path):
    xydt = 1.0
    wdt = 1.0
    xRobotPath = []
    yRobotPath = []
    prefHeadingX = []
    prefHeadingY = []
    currHeadingX = []
    currHeadingY = []
    tangentVecNorms = []

    pose = startPose
    sumDisplacement = 0
    steps = 60
    lastVec = Vector()
    iters = 0
    while not controller.isFinished:
        iters+=1
        output = controller.update(pose)[0]
        displacement: Pose = Pose(Vector(output.x * xydt, output.y * xydt), output.heading * wdt)
        sumDisplacement += displacement.vec.norm()

        pose = Pose(Vector(pose.x + displacement.x, pose.y + displacement.y), angleNorm(pose.heading + radians(displacement.heading)))

        displacement = lastVec.minus(pose.vec).norm()
        
        if displacement > path.length() / steps:
            xRobotPath.append(pose.x)
            yRobotPath.append(pose.y)
            prefHeading = output.vec.angle()
            currHeading = pose.heading
            prefHeadingVec = Vector.polar(1, prefHeading)
            currHeadingVec = Vector.polar(1, currHeading)
            prefHeadingX.append(prefHeadingVec.x)
            prefHeadingY.append(prefHeadingVec.y)
            currHeadingX.append(currHeadingVec.x)
            currHeadingY.append(currHeadingVec.y)
            lastVec = pose.vec
            tangentVecNorms.append(output.vec.norm())


    print(iters)

    plt.quiver(xRobotPath, yRobotPath, prefHeadingX, prefHeadingY, color='r', scale_units='inches', scale=5)
    plt.quiver(xRobotPath, yRobotPath, currHeadingX, currHeadingY, color='c', scale_units='inches', scale=7)

def plot(pose, path, controller):
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    plotPath(path)
    plotRobot(pose, controller, path)
    plotKnots(path)
    plt.show()

def plotCircle(pos, radius, c_color):
    theta = np.linspace(0, 2*pi, 100)
    cx = radius * np.cos(theta) + pos.x
    cy = radius * np.sin(theta) + pos.y
    cx, cy = list(cx), list(cy)
    fit_c_x = []
    fit_c_y = []
    for i,x in enumerate(cx):
        y = cy[i]
        # in_x_bounds = minBounds.x < x < maxBounds.x
        # in_y_bounds = minBounds.y < y < maxBounds.y
        # if in_x_bounds and in_y_bounds:
        fit_c_x.append(x)
        fit_c_y.append(y)
    plt.plot(fit_c_x, fit_c_y, color=c_color)

def plotKnots(path: Path):
    plotCircle(path.segments[0].start().vec, 1, 'w')
    for segment in path.segments:
        plotCircle(segment.end().vec, 1, 'w')

def path_generator(start: Pose, *poses) -> Path:
    builder = PathBuilder(start, start.heading)
    for pose in poses:
        vec = pose.vec
        angle = pose.heading
        builder = builder.splineTo(vec, angle)
    return builder.build()










