from math import radians
import numpy as np
from path import *
from geometry import * 

def main():
    pose = Pose(Vector(5.0, 2.0), radians(90.0))
    path: Path = PathBuilder(Pose(Vector(), radians(90.0)), radians(90.0)).splineTo(Vector(24.0, 24.0), radians(70.0)).splineTo(Vector(48.0, 36.0), 0.0).build()
    len = path.length()

    controller = gvf(path, 0.25, 1.0, 5.0, 2.0)
    xRobot = []
    yRobot = []
    xHeading = []
    yHeading = []
    dt = 1.0

    xs = np.arange(0.0, 50.0, 1.5)
    ys = np.arange(0.0, 40.0, 1.5)
    for x in xs:
        for y in ys:
            pose = Pose(Vector(float(x), float(y)), 0.0)
            output = controller.update(pose)
            xRobot.append(pose.x)
            yRobot.append(pose.y)
            xHeading.append(output.x)
            yHeading.append(output.y)


    # while not controller.finished:
    #     output = controller.update(pose)
    #     displacement: Pose = output[0].times(dt)
    #     pose = Pose(Vector(pose.x + displacement.x, pose.y + displacement.y), angleNorm(pose.heading + displacement.heading))
    #     xRobot.append(pose.x)
    #     yRobot.append(pose.y)
    #     normed = output[1].normalized()
    #     xHeading.append(normed.x)
    #     yHeading.append(normed.y)

    xPath, yPath = [], []
    displacements = np.arange(0, len)
    for s in displacements:
        v = path.get(float(s))
        xPath.append(v.x)
        yPath.append(v.y)

    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (10,8)
    # plt.plot(xRobot,yRobot, color='r', linewidth=3, alpha=0.9)
    plt.plot(xPath,yPath, color='w', linewidth=3, alpha=0.5)
    plt.quiver(xRobot, yRobot, xHeading, yHeading, color='c', scale_units='inches', scale=5)
    plt.show()

if __name__ == '__main__':
    main()