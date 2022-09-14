from math import radians
from geometry import * 
from koawalib import *
from plothelper import *

def main():
    pose = Pose(Vector(-58, -36), 0)
    path: Path = path_generator(pose, 
            Pose(Vector(-54, -54), radians(330)),
            Pose(Vector(-36, -54), radians(0)),
            Pose(Vector(-12, -58), radians(330)))

    controller = koawalib_gvf(path, kN=0.5, kOmega=1.0, kF=4.0, epsilon=0.4)
    plot(pose, path, controller)

if __name__ == '__main__':
    main()
