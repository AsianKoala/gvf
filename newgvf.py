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
    # plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.8)

def plotRobot(startPose, controller: koawalib_gvf, path: Path):
    xydt = 2.0
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
    steps = 50
    lastVec = Vector()
    while not controller.isFinished:
        output = controller.update(pose)[0]
        displacement: Pose = Pose(Vector(output.x * xydt, output.y * xydt), output.heading * wdt)
        sumDisplacement += displacement.vec.norm()

        pose = Pose(Vector(pose.x + displacement.x, pose.y + displacement.y), angleNorm(pose.heading + radians(displacement.heading)))

        displacement = lastVec.minus(pose.vec).norm()
        
        # this is only for graphing
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

        # print('\n')

    # plt.plot(xRobotPath,yRobotPath, color='r', linewidth=3, alpha=0.6)
    plt.quiver(xRobotPath, yRobotPath, prefHeadingX, prefHeadingY, color='r', scale_units='inches', scale=5)
    plt.quiver(xRobotPath, yRobotPath, currHeadingX, currHeadingY, color='c', scale_units='inches', scale=7)



def main():
    pose = Pose(Vector(0.0, 0.0), radians(90.0))
    path: Path = PathBuilder(pose, pose.heading).splineTo(Vector(24,24), radians(90.0)).build()
    # kN is normal vector weight
    # kOmega is curvature weight
    # kF is ending weight
    # epsilon is end goal distance
    controller = koawalib_gvf(path, kN=0.005, kOmega=2.0, kF=4.0, epsilon=0.4)
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    plotPath(path)
    plotRobot(pose, controller, path)
    plt.show()

if __name__ == '__main__':
    main()
