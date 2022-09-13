from math import radians
import numpy as np
from path import *
from geometry import * 

def plotPath(path):
    len = path.length()
    xPath, yPath = [], []
    displacements = np.linspace(0.0, len, num=int(len*2))
    for s in displacements:
        v = path.get(float(s))
        xPath.append(v.x)
        yPath.append(v.y)
    plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.3)

def plotRobot(startPose, controller: gvf, path: Path):
    xydt = 1.0
    wdt = 1.0
    xRobotPath = []
    yRobotPath = []
    prefHeadingX = []
    prefHeadingY = []
    currHeadingX = []
    currHeadingY = []

    pose = startPose
    sumDisplacement = 0
    steps = 50
    lastVec = Vector()
    while not controller.finished:
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

    # plt.plot(xRobotPath,yRobotPath, color='r', linewidth=3, alpha=0.6)
    # plt.quiver(xRobotPath, yRobotPath, prefHeadingX, prefHeadingY, color='r', scale_units='inches', scale=5)
    plt.quiver(xRobotPath, yRobotPath, currHeadingX, currHeadingY, color='c', scale_units='inches', scale=7)


def plotVectorField(controller:gvf, minBounds, maxBounds):
    controller.print = False
    xRobot = []
    yRobot = []
    xHeading = []
    yHeading = []
    xs = np.arange(minBounds.x, maxBounds.x, 1.0)
    ys = np.arange(minBounds.y, maxBounds.y, 1.0)
    for x in xs:
        for y in ys:
            pose = Pose(Vector(float(x), float(y)), 0.0)
            output = controller.update(pose)[0]
            xRobot.append(pose.x)
            yRobot.append(pose.y)
            xHeading.append(output.x)
            yHeading.append(output.y)
    plt.quiver(xRobot, yRobot, xHeading, yHeading, color='c', scale_units='inches', scale=7)

def plotCircle(pos, radius, c_color, minBounds, maxBounds):
    theta = np.linspace(0, 2*pi, 100)
    cx = radius * np.cos(theta) + pos.x
    cy = radius * np.sin(theta) + pos.y
    cx, cy = list(cx), list(cy)
    fit_c_x = []
    fit_c_y = []
    for i,x in enumerate(cx):
        y = cy[i]
        in_x_bounds = minBounds.x < x < maxBounds.x
        in_y_bounds = minBounds.y < y < maxBounds.y
        if in_x_bounds and in_y_bounds:
            fit_c_x.append(x)
            fit_c_y.append(y)
    plt.plot(fit_c_x, fit_c_y, color=c_color)
            
def main():
    pose = Pose(Vector(-70, -24), 0)
    stack = Pose(Vector(-12, -70), radians(90.0))
    rVec = stack.vec.minus(pose.vec)
    # path: Path = PathBuilder(pose, pose.heading).splineTo(Vector(24,24), radians(90.0)).build()
    path: Path = PathBuilder(pose, pose.heading).splineTo(rVec, stack.heading).build()

    controller = gvf(path, kN = 0.1, kOmega = 1.0, kTheta = 50.0, kF = 4.0, kEnd = 0.4)
    minBounds = Vector(-5, -5)
    maxBounds = Vector(25, 25)
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    plotPath(path)
    plotRobot(pose, controller, path)
    # plotVectorField(controller, minBounds, maxBounds)
    # plotCircle(path.end().vec, controller.kF, 'y', minBounds, maxBounds)
    # plotCircle(path.end().vec, controller.kEnd, 'w', minBounds, maxBounds)
    plt.show()

if __name__ == '__main__':
    main()
