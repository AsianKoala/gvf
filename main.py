from math import radians
import numpy as np
from path import *
from geometry import * 

def plotPath(path):
    len = path.length()
    xPath, yPath = [], []
    displacements = np.arange(0.0, len)
    for s in displacements:
        v = path.get(float(s))
        xPath.append(v.x)
        yPath.append(v.y)
    plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.8)

def plotRobot(startPose, controller):
    dt = 1.0
    xRobotPath = []
    yRobotPath = []
    pose = startPose
    while not controller.finished:
        output = controller.update(pose)
        displacement: Pose = output.times(dt)
        pose = Pose(Vector(pose.x + displacement.x, pose.y + displacement.y), angleNorm(pose.heading + displacement.heading))
        xRobotPath.append(pose.x)
        yRobotPath.append(pose.y)
    plt.plot(xRobotPath,yRobotPath, color='r', linewidth=3, alpha=0.9)
    
def plotVectorField(controller, minBounds, maxBounds):
    xRobot = []
    yRobot = []
    xHeading = []
    yHeading = []
    xs = np.arange(minBounds.x, maxBounds.x, 1.5)
    ys = np.arange(minBounds.y, maxBounds.y, 1.5)
    for x in xs:
        for y in ys:
            pose = Pose(Vector(float(x), float(y)), 0.0)
            output = controller.update(pose)
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
    pose = Pose(Vector(10.0, 2.0), radians(90.0))
    startPose = Pose(pose.vec, pose.heading)
    path: Path = PathBuilder(Pose(Vector(), radians(90.0)), radians(70.0)).splineTo(
        Vector(24.0, 24.0), radians(70.0)).splineTo(
            Vector(48.0, 36.0), 0.0).splineTo(Vector(64.0, 12.0), radians(-70)).build()

    controller = gvf(path, 0.2, 1.0, 6.0, 4.0)
    minBounds = Vector(-5, -5)
    maxBounds = Vector(70, 40)
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    plotPath(path)
    plotRobot(startPose, controller)
    plotVectorField(controller, minBounds, maxBounds)
    plotCircle(path.end().vec, controller.kF, 'y', minBounds, maxBounds)
    # plotCircle(path.end().vec, controller.kEnd, 'b', minBounds, maxBounds)
    plt.show()

if __name__ == '__main__':
    main()
