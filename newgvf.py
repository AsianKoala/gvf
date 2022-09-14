from math import radians
from geometry import * 
from koawalib import *
from plothelper import *

def main():
    init_plot()

    path(Pose(Vector(-70, -30), 0),
        Pose(Vector(-56, -60), radians(330)), 
        Pose(Vector(-12, -70), radians(330)))

    path(Pose(Vector(-12, -70), radians(150)),
        Pose(Vector(-3, -30), radians(65)))

    path(Pose(Vector(-3, -30), radians(245)),
        Pose(Vector(-12, -68), radians(270)))

    finish_plot()

if __name__ == '__main__':
    main()
