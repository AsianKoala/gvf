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
    plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.5)

def plotRobot(path, startPose, controller):
    dt = 1.0
    xRobotPath = []
    yRobotPath = []
    pose = startPose
    controller.canFinish = True
    controller.finished = False
    while not controller.finished:
        output = controller.update(pose)
        displacement: Pose = output.times(dt)
        pose = Pose(Vector(pose.x + displacement.x, pose.y + displacement.y), angleNorm(pose.heading + displacement.heading))
        xRobotPath.append(pose.x)
        yRobotPath.append(pose.y)
    plt.plot(xRobotPath,yRobotPath, color='r', linewidth=3, alpha=0.9)
    
def plotVectorField(controller):
    xRobot = []
    yRobot = []
    xHeading = []
    yHeading = []
    xs = np.arange(0.0, 64.0, 1.5)
    ys = np.arange(-5.0, 40.0, 1.5)
    controller.canFinish = False
    controller.finished = False
    for x in xs:
        for y in ys:
            pose = Pose(Vector(float(x), float(y)), 0.0)
            output = controller.update(pose)
            xRobot.append(pose.x)
            yRobot.append(pose.y)
            xHeading.append(output.x)
            yHeading.append(output.y)
    plt.quiver(xRobot, yRobot, xHeading, yHeading, color='c', scale_units='inches', scale=7)



def main():
    pose = Pose(Vector(10.0, -4.0), radians(90.0))
    startPose = Pose(pose.vec, pose.heading)
    path: Path = PathBuilder(Pose(Vector(), radians(90.0)), radians(90.0)).splineTo(
        Vector(24.0, 24.0), radians(70.0)).splineTo(
            Vector(48.0, 36.0), 0.0).splineTo(Vector(64.0, 12.0), radians(-90)).build()

    controller = gvf(path, 0.2, 1.0, 5.0, 2.0)

    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    plotPath(path)
    plotRobot(path, startPose, controller)
    plotVectorField(controller)
    plt.show()

if __name__ == '__main__':
    main()
