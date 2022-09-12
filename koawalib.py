from typing import List
import numpy as np
from numpy import sign, pi
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import degrees, radians

from scipy.misc import derivative
from geometry import Pose, Vector, epsilonEquals, angleNorm
from path import Path 

def linear_error_map(input):
    return input

class koawalib_gvf:
    def __init__(self, path: Path, kN, kOmega, kF, epsilon):
        self.path = path
        self.kN = kN
        self.kOmega = kOmega
        self.kF = kF
        self.epsilon = epsilon
        self.errorMap = linear_error_map
        self.lastPose: Pose = Pose
        self.lastS = None
        self.lastGVFVec = Vector()
        self.lastHeadingError = 0.0
        self.isFinished = False

    def gvfVecAt(self, pose: Pose, s) -> Vector:
        tangentVec = self.path.deriv(s).vec
        normalVec = tangentVec.rotate(pi / 2.0)
        projected = self.path.get(s).vec
        displacementVec = projected.minus(pose.vec)
        orientation = displacementVec.cross(tangentVec)
        error = displacementVec.norm() * sign(orientation)
        res = tangentVec.minus(normalVec.times(self.kN * self.errorMap(error)))
        # print('res:', res)
        # print('tangent vec:', tangentVec)
        print('error:', error)
        return res

    def headingControl(self):
        desiredHeading = self.lastGVFVec.angle()
        headingError = angleNorm(desiredHeading - self.lastPose.heading)
        result = self.kOmega * headingError
        # print('heading error:', headingError)
        # print('turning output:', result)
        # return result, headingError
        return 0, headingError

    def vectorControl(self):
        projectedDisplacement = abs(self.lastS - self.path.length())
        # endDisplacement = abs(self.lastS - self.path.length())
        translationalPower = self.lastGVFVec.times(projectedDisplacement / self.kF)
        absoluteDisplacement = self.path.end().vec.minus(self.lastPose.vec)
        self.isFinished = projectedDisplacement < self.epsilon and absoluteDisplacement.norm() < self.epsilon
        absoluteVector = absoluteDisplacement.times(projectedDisplacement / self.kF)
        if self.isFinished: translationalPower = absoluteVector
        if translationalPower.norm() > 1.0: translationalPower = translationalPower.normalized()
        return translationalPower

    def update(self, currPose):
        self.lastPose = currPose
        if self.lastS == None:
            self.lastS = self.path.fastProject(self.lastPose.vec, self.path.length() * 0.25)
        else:
            self.lastS = self.path.fastProject(self.lastPose.vec, self.lastS)

        vectorFieldResult: Vector = self.gvfVecAt(self.lastPose, self.lastS)
        self.lastGVFVec = vectorFieldResult
        angularOutput, headingError = self.headingControl()
        self.lastHeadingError = headingError
        vectorResult = self.vectorControl()
        # rotated is just field oriented vectorResult (i think)
        rotated = vectorResult.rotate(pi / 2.0 - self.lastPose.heading)
        return Pose(vectorResult, angularOutput), rotated











