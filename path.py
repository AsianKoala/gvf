from typing import List
import numpy as np
from numpy import sign, pi
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import radians
from geometry import Pose, Vector, epsilonEquals, angleNorm

class QuinticPolynomial:
    COEFF_MATRIX = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        [20.0, 12.0, 6.0, 2.0, 0.0, 0.0]
    ])

    def __init__(self, start, startDeriv, startSecondDeriv, end, endDeriv, endSecondDeriv):
        target = np.array([
            [start],
            [startDeriv],
            [startSecondDeriv],
            [end],
            [endDeriv],
            [endSecondDeriv]
        ])

        solved = la.solve(self.COEFF_MATRIX, target)
        self.a = float(solved[0])
        self.b = float(solved[1])
        self.c = float(solved[2])
        self.d = float(solved[3])
        self.e = float(solved[4])
        self.f = float(solved[5])

    def get(self, t): return (self.a * t + self.b) * (t * t * t * t) + self.c * (t * t * t) + self.d * (t * t) + self.e * t + self.f

    def deriv(self, t): return (5 * self.a * t + 4 * self.b) * (t * t * t) + (3 * self.c * t + 2 * self.d) * t + self.e

    def secondDeriv(self, t): return (20 * self.a * t + 12 * self.b) * (t * t) + 6 * self.c * t + 2 * self.d

    def thirdDeriv(self, t): return (60 * self.a * t + 24 * self.b) * t + 6 * self.c

    def __str__(self): "{}*t^5+{}*t^4+{}*t^3+{}*t^2+{}*t+{}".format(self.a, self.b, self.c, self.d, self.e, self.f)

class ParametricCurve:
    def length(self): pass

    def reparam(self, s): pass
    def internalGet(self, t) -> Vector: pass
    def internalDeriv(self, t) -> Vector: pass
    def internalSecondDeriv(self, t) -> Vector: pass
    def internalThirdDeriv(self, t) -> Vector: pass

    def paramDeriv(self, t): pass
    def paramSecondDeriv(self, t): pass
    def paramThirdDeriv(self, t): pass

    def get(self, s, t = None) -> Vector:
        realT = t
        if t == None: realT = self.reparam(s)
        return self.internalGet(realT)

    def deriv(self, s, t = None) -> Vector:
        realT = t
        if t == None: realT = self.reparam(s)
        return self.internalDeriv(realT).times(self.paramDeriv(realT))

    def secondDeriv(self, s, t = None) -> Vector:
        realT = t
        if t == None: realT = self.reparam(s)
        deriv = self.internalDeriv(realT)
        secondDeriv = self.internalSecondDeriv(realT)
        paramDeriv = self.paramDeriv(realT)
        paramSecondDeriv = self.paramSecondDeriv(realT)
        return secondDeriv.times(paramDeriv * paramDeriv).plus(deriv.times(paramSecondDeriv))

    def thirdDeriv(self, s, t = None) -> Vector:
        realT = t
        if t == None: realT = self.reparam(s)
        deriv = self.internalDeriv(realT)
        secondDeriv = self.internalSecondDeriv(realT)
        thirdDeriv = self.internalThirdDeriv(realT)
        paramDeriv = self.paramDeriv(realT)
        paramSecondDeriv = self.paramSecondDeriv(realT)
        paramThirdDeriv = self.paramThirdDeriv(realT)
        return thirdDeriv * paramDeriv * paramDeriv * paramDeriv + secondDeriv * paramSecondDeriv * paramDeriv * 3.0 + deriv * paramThirdDeriv

    def start(self): return self.get(0.0, 0.0)
    def startDeriv(self): return self.deriv(0.0, 0.0)
    def startSecondDeriv(self): return self.secondDeriv(0.0, 0.0)
    def startThirdDeriv(self): return self.thirdDeriv(0.0, 0.0)
    def end(self): return self.get(self.length(), 1.0)
    def endDeriv(self): return self.deriv(self.length(), 1.0)
    def endSecondDeriv(self): return self.secondDeriv(self.length(), 1.0)
    def endThirdDeriv(self): return self.thirdDeriv(self.length(), 1.0)
    
    def tangentAngle(self, s, t = None):
        realT = t
        if t == None: realT = self.reparam(s)
        return self.deriv(s, realT).angle()

    def tangentAngleDeriv(self, s, t = None):
        realT = t
        if t == None: realT = self.reparam(s)
        deriv = self.deriv(s, realT)
        secondDeriv = self.secondDeriv(s, realT)
        return deriv.cross(secondDeriv)

    def tangentAngleSecondDeriv(self, s, t = None):
        realT = t
        if t == None: realT = self.reparam(s)
        deriv = self.deriv(s, realT)
        thirdDeriv = self.thirdDeriv(s, realT)
        return deriv.cross(thirdDeriv)

class Knot:
    def __init__(self, pos, deriv=Vector(), secondDeriv=Vector()):
        self.x = pos.x
        self.y = pos.y
        self.dx = deriv.x
        self.dy = deriv.y
        self.d2x = secondDeriv.x
        self.d2y = secondDeriv.y
        self.pos = pos
        self.deriv = deriv
        self.secondDeriv = secondDeriv
    
    @classmethod
    def fromNums(x, y, dx, dy, d2x, d2y): return Knot(Vector(x,y), Vector(dx,dy), Vector(d2x, d2y))

class HeadingInterpolator:
    def init(self, curve: ParametricCurve):
        self.curve = curve

    def get(self, s, t = None):
        realT = t
        if t == None: realT = self.curve.reparam(s)
        return self.internalGet(s, realT)

    def deriv(self, s, t = None):
        realT = t
        if t == None: realT = self.curve.reparam(s)
        return self.internalDeriv(s, realT)
    
    def secondDeriv(self, s, t = None):
        realT = t
        if t == None: realT = self.curve.reparam(s)
        return self.internalSecondDeriv(s, realT)
    
    def start(self): return self.get(0,0)
    def startDeriv(self): return self.deriv(0,0)
    def startSecondDeriv(self): return self.secondDeriv(0,0)
    def end(self): return self.get(self.curve.length(), 1.0)
    def endDeriv(self): return self.deriv(self.curve.length(), 1.0)
    def endSecondDeriv(self): return self.secondDeriv(self.curve.length(), 1.0)

    def internalGet(self, s, t): pass
    def internalDeriv(self, s, t): pass
    def internalSecondDeriv(self, s, t): pass

class ConstantInterpolator(HeadingInterpolator):
    def __init__(self, heading):
        self.heading = heading

    def internalGet(self, s, t): return angleNorm(self.heading)
    def internalDeriv(self, s, t): return 0.0
    def internalSecondDeriv(self, s, t): return 0.0

class LinearInterpolator(HeadingInterpolator):
    def __init__(self, startHeading, angle):
        self.startHeading = startHeading
        self.angle = angle

    def internalGet(self, s, t): return angleNorm(self.startHeading + s / self.curve.length() * self.angle)
    def internalDeriv(self, s, t): return self.angle / self.curve.length()
    def internalSecondDeriv(self, s, t): 0.0

class TangentInterpolator(HeadingInterpolator):
    def __init__(self, offset = 0.0):
        self.offset = offset
    
    def internalGet(self, s, t): return float(angleNorm(self.offset + self.curve.tangentAngle(s, t)))
    def internalDeriv(self, s, t): return float(self.curve.tangentAngleDeriv(s, t))
    def internalSecondDeriv(self, s, t): return float(self.curve.tangentAngleSecondDeriv(s, t))

class SplineInterpolator(HeadingInterpolator):
    def __init__(self, startHeading, endHeading, startHeadingDeriv = None, startHeadingSecondDeriv = None, endHeadingDeriv = None, endHeadingSecondDeriv = None):
        self.startHeading = startHeading
        self.endHeading = endHeading
        self.startHeadingDeriv = startHeadingDeriv
        self.startHeadingSecondDeriv = startHeadingSecondDeriv
        self.endHeadingDeriv = endHeadingDeriv
        self.endHeadingSecondDeriv = endHeadingSecondDeriv
        self.tangentInterpolator = TangentInterpolator()
        self.headingSpline = None

    
    def init(self, curve: ParametricCurve):
        super().init(curve)
        self.tangentInterpolator.init(self.curve)
        len = self.curve.length()
        headingDelta = angleNorm(self.endHeading - self.startHeading)

        if self.startHeadingDeriv == None: self.startHeadingDeriv = self.curve.tangentAngleDeriv(0,0)
        if self.startHeadingSecondDeriv == None: self.startHeadingSecondDeriv = self.curve.tangentAngleSecondDeriv(0,0)
        if self.endHeadingDeriv == None: self.endHeadingDeriv = self.curve.tangentAngleDeriv(len,1.0)
        if self.endHeadingSecondDeriv == None: self.endHeadingSecondDeriv = self.curve.tangentAngleSecondDeriv(len,1.0)

        self.headingSpline = QuinticPolynomial(
            0.0,
            self.startHeadingDeriv * len,
            self.startHeadingSecondDeriv * len * len,
            headingDelta,
            self.endHeadingDeriv * len,
            self.endHeadingSecondDeriv * len * len
        )

    def internalGet(self, s, t): return angleNorm(self.startHeading + self.headingSpline.get(s / self.curve.length()))
    def internalDeriv(self, s, t):
        len = self.curve.length()
        return self.headingSpline.deriv(s / len) / len
    def internalSecondDeriv(self, s, t):
        len = self.curve.length()
        return self.headingSpline.secondDeriv(s / len) / (len * len)

class QuinticSpline(ParametricCurve):
    def __init__(self, start: Knot, end: Knot, maxDeltaK=0.01, maxSegmentLength=0.25, maxDepth=30):
        self.x = QuinticPolynomial(start.x, start.dx, start.d2x, end.x, end.dx, end.d2x)
        self.y = QuinticPolynomial(start.y, start.dy, start.d2y, end.y, end.dy, end.d2y)
        self.splineLength = 0.0
        self.sSamples = []
        self.tSamples = []
        self.maxDeltaK = maxDeltaK
        self.maxSegmentLength = maxSegmentLength
        self.maxDepth = maxDepth
        self.parameterize(0.0, 1.0)

    def approxLength(self, v1: Vector, v2: Vector, v3: Vector):
        w1 = (v2.minus(v1)).times(2.0)
        w2 = (v2.minus(v3)).times(2.0)
        det = w1.cross(w2)
        chord = v1.dist(v3)
        if epsilonEquals(det, 0.0):
            return chord
        else:
            x1 = v1.dot(v1)
            x2 = v2.dot(v2)
            x3 = v3.dot(v3)

            y1 = x2 - x1
            y2 = x2 - x3

            origin = Vector(y1 * w2.y - y2 * w1.y, y2 * w1.x - y1 * w2.x).div(det)
            radius = origin.dist(v1)
            return 2.0 * radius * np.arcsin(chord / (2.0 * radius))

    def internalCurvature(self, t):
        deriv = self.internalDeriv(t)
        derivNorm = deriv.norm()
        secondDeriv = self.internalSecondDeriv(t)
        if epsilonEquals(derivNorm, 0.0): return 0
        return abs(secondDeriv.cross(deriv) / (derivNorm ** 3))

    def parameterize(self, tLo, tHi, vLo = None, vHi = None, depth = 0):
        if vLo == None: vLo = self.internalGet(tLo)
        if vHi == None: vHi = self.internalGet(tHi)
        
        tMid = 0.5 * (tLo + tHi)
        vMid = self.internalGet(tMid)

        deltaK = abs(self.internalCurvature(tLo) - self.internalCurvature(tHi))
        segmentLength = self.approxLength(vLo, vMid, vHi)
        if segmentLength == None: print('asfafafasfaf')

        if depth < self.maxDepth and (deltaK > self.maxDeltaK or segmentLength > self.maxSegmentLength):
            self.parameterize(tLo, tMid, vLo, vMid, depth + 1)
            self.parameterize(tMid, tHi, vMid, vHi, depth + 1)
        else:
            self.splineLength += segmentLength
            self.sSamples.append(self.splineLength)
            self.tSamples.append(tHi)

    
    def internalGet(self, t): return Vector(float(self.x.get(t)), float(self.y.get(t)))
    def internalDeriv(self, t): return Vector(float(self.x.deriv(t)), float(self.y.deriv(t)))
    def internalSecondDeriv(self, t): return Vector(float(self.x.secondDeriv(t)), float(self.y.secondDeriv(t)))
    def internalThirdDeriv(self, t): return Vector(float(self.x.thirdDeriv(t)), float(self.y.thirdDeriv(t)))

    def interp(self, s, sLo, sHi, tLo, tHi): return tLo + (s - sLo) * (tHi - tLo) / (sHi - sLo)

    def reparam(self, s):
        if s <= 0.0: return 0.0
        if s >= self.splineLength: return 1.0

        lo = 0
        hi = 0
        for x in self.sSamples:
            hi+=1

        while lo <= hi:
            mid = (hi + lo) // 2 

            if s < self.sSamples[mid]:
                hi = mid - 1
            elif s > self.sSamples[mid]:
                lo = mid + 1
            else:
                return self.tSamples[mid]

        return self.interp(s, self.sSamples[lo], self.sSamples[hi], self.tSamples[lo], self.tSamples[hi])

    def paramDeriv(self, t):
        deriv = self.internalDeriv(t)
        return 1.0 / np.sqrt(deriv.dot(deriv))

    def paramSecondDeriv(self, t):
        deriv = self.internalDeriv(t)
        secondDeriv = self.internalSecondDeriv(t)
        numerator = -(deriv.dot(secondDeriv))
        denominator = deriv.dot(deriv)
        return numerator / (denominator ** 2)

    def paramThirdDeriv(self, t):
        deriv = self.internalDeriv(t)
        secondDeriv = self.secondDeriv(t)
        thirdDeriv = self.thirdDeriv(t)

        firstNumeratorInput = 2.0 * deriv.dot(secondDeriv)
        secondNumerator = secondDeriv.dot(secondDeriv) + deriv.dot(thirdDeriv)

        denominator = deriv.dot(deriv)
        return firstNumeratorInput * firstNumeratorInput / np.power(denominator, 3.5) -  secondNumerator / np.power(denominator, 2.5)

    def length(self): return self.splineLength
 

class PathSegment:
    def __init__(self, curve: ParametricCurve, interpolator: HeadingInterpolator = TangentInterpolator()):
        self.curve = curve
        self.interpolator = interpolator
        self.interpolator.init(self.curve)

    def length(self): return self.curve.length()

    def get(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return Pose(self.curve.get(s, realT), self.interpolator.get(s, realT))

    def deriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return Pose(self.curve.deriv(s, realT), self.interpolator.deriv(s, realT))

    def secondDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return Pose(self.curve.secondDeriv(s, realT), self.interpolator.secondDeriv(s, realT))

    def tangentAngle(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return self.curve.tangentAngle(s, realT)
    
    def internalDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return Pose(self.curve.internalDeriv(realT), self.interpolator.internalDeriv(s, realT))

    def internalSecondDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        return Pose(self.curve.internalSecondDeriv(realT), self.interpolator.internalSecondDeriv(s, realT))
    
    def reparam(self, s): return self.curve.reparam(s)

    def start(self): return self.get(0)
    def startDeriv(self): return self.deriv(0)
    def startSecondDeriv(self): return self.secondDeriv(0)
    def startTangentAngle(self): return self.tangentAngle(0)
    def startInternalDeriv(self): return self.internalDeriv(0)
    def startInternalSecondDeriv(self): return self.internalSecondDeriv(0)

    def end(self): return self.get(self.length())
    def endDeriv(self): return self.deriv(self.length())
    def endSecondDeriv(self): return self.secondDeriv(self.length())
    def endTangentAngle(self): return self.tangentAngle(self.length())
    def endInternalDeriv(self): return self.internalDeriv(self.length())
    def endInternalSecondDeriv(self): return self.internalSecondDeriv(self.length())

class Path:
    def __init__(self, segments: List[PathSegment]):
        self.segments = segments

    def length(self):
        sum = 0
        for segment in self.segments:
            sum += segment.length()
        return sum
    
    def segment(self, s):
        if s<=0:
            return [self.segments[0], 0]
        remainingDisplacement = s
        for segment in self.segments:
            if remainingDisplacement <= segment.length():
                return [segment, remainingDisplacement]
            remainingDisplacement -= segment.length()
        last = self.segments[-1]
        return [last, last.length()]

    def get(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.get(remainingDisplacement, realT)
    
    def deriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.deriv(remainingDisplacement, realT)

    def secondDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.secondDeriv(remainingDisplacement, realT)

    def internalDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.internalDeriv(remainingDisplacement, realT)

    def internalSecondDeriv(self, s, t=None):
        realT = t
        if t == None: realT = self.reparam(s)
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.internalSecondDeriv(remainingDisplacement, realT)

    def reparam(self, s):
        pair = self.segment(s)
        segment, remainingDisplacement = pair[0], pair[1]
        return segment.reparam(remainingDisplacement)

    
    def fastProject(self, query, pGuess = None):
        projectGuess = pGuess
        if pGuess == None: projectGuess = self.length() / 2.0
        s = projectGuess
        for _ in range(200):
            t = self.reparam(s)
            pathPoint = self.get(s, t).vec
            deriv = self.deriv(s,t).vec
            ds = (query.minus(pathPoint)).dot(deriv)
            if epsilonEquals(ds, 0.0): break
            s += ds
            if s<= 0.0: break
            if s>= self.length(): break
        return max(0.0, min(s, self.length()))

    def start(self): return self.get(0)
    def startDeriv(self): return self.startDeriv(0)
    def startSecondDeriv(self): return self.startSecondDeriv(0)
    def end(self): return self.end(0)
    def endDeriv(self): return self.endDeriv(0)
    def endSecondDeriv(self): return self.endSecondDeriv(0)

class PathBuilder:
    def __init__(self, startPose, startTangent):
        self.startPose: Pose = startPose
        if startTangent == None:
            self.startTangent = self.startPose.heading
        else:
            self.startTangent = startTangent
        self.path: Path = None
        self.s = None
        self.currentPose: Pose = startPose
        self.currentTangent = self.startTangent
        self.segments: List[PathSegment] = []
        
    def makeSpline(self, endPosition: Vector, endTangent):
        startPose = None
        if self.currentPose == None:
            startPose = self.path.get(self.s)
        else:
            startPose = self.currentPose
        derivMag = startPose.vec.dist(endPosition)
        startWaypoint = None
        endWaypoint = Knot(endPosition, Vector.polar(derivMag, endTangent))
        if self.currentPose == None:
            startDeriv = self.path.internalDeriv(self.s).vec
            startSecondDeriv = self.path.internalSecondDeriv(self.s).vec
            startWaypoint = Knot(startPose.vec, startDeriv, startSecondDeriv)
        else:
            startWaypoint = Knot(startPose.vec, Vector.polar(derivMag, self.currentTangent))
        return QuinticSpline(startWaypoint, endWaypoint)    

    def makeTangentInterpolator(self, curve: ParametricCurve):
        if self.currentPose == None:
            prevSegment: PathSegment = self.path.segment(self.s)[0]
            prevInterpolator: TangentInterpolator = prevSegment.interpolator
            return TangentInterpolator(prevInterpolator.offset)
        
        startHeading = curve.tangentAngle(0, 0)
        interpolator = TangentInterpolator(self.currentPose.heading - startHeading)
        interpolator.init(curve)
        return interpolator

    def addSegment(self, segment: PathSegment):
        self.currentPose = segment.end()
        self.currentTangent = segment.endTangentAngle()
        self.segments.append(segment)
        return self

    def splineTo(self, endPosition: Vector, endTangent):
        spline = self.makeSpline(endPosition, endTangent)
        interpolator = self.makeTangentInterpolator(spline)
        return self.addSegment(PathSegment(spline, interpolator))

    def build(self):
        return Path(self.segments)
    
def linear(input): return input

class gvf:
    def __init__(self, path: Path, kN, kOmega, kF=None, epsilon = 1.0, errorMap = linear):
        self.path = path
        self.kN = kN
        self.kOmega = kOmega
        self.kF = kF
        self.epsilon = epsilon
        self.errorMap = errorMap
        self.lastS = None
        self.finished = False
        self.canFinish = True

    def update(self, pose: Pose):
        if self.lastS == None:
            self.lastS = self.path.fastProject(pose.vec, self.path.length() * 0.25)
        else:
            self.lastS = self.path.fastProject(pose.vec, self.lastS)

        s = self.lastS
        tangent: Vector = self.path.deriv(s).vec
        normalVec = tangent.rotate(pi / 2.0)

        projected: Pose = self.path.get(s).vec
        displacementVec = projected.minus(pose.vec)
        orientation = displacementVec.cross(tangent)
        error = displacementVec.norm() * sign(orientation)

        vectorFieldResult = tangent.minus(normalVec.times(self.kN * self.errorMap(error)))

        desiredHeading = vectorFieldResult.angle()
        headingError = angleNorm(desiredHeading - pose.heading)
        angularOutput = self.kOmega * headingError

        endDisplacement = abs(s - self.path.length())
        forwardOutput = None
        if self.kF == None: forwardOutput = 1.0
        else: forwardOutput = endDisplacement / self.kF
        self.finished = self.finished or (self.canFinish and endDisplacement < self.epsilon)
        translationalPower = vectorFieldResult
        if self.finished: translationalPower = (projected.minus(pose.vec))
        translationalPower = translationalPower.times(forwardOutput)
        if translationalPower.norm() > 1.0:
            translationalPower = translationalPower.normalized()

        return Pose(translationalPower, angularOutput)


