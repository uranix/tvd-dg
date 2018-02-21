import math
import numpy as np
from problem import Problem

from param import Param

class Hopf(Problem):
    @staticmethod
    def u0(x):
        return [0.5 + np.sin(2 * np.pi * x)]

    @staticmethod
    def p(x):
        return Param(1, 1)

    @staticmethod
    def F(u, p):
        return 0.5 * u * u

    @staticmethod
    def T():
        return 0.3

    @staticmethod
    def L():
        return 1.0

    @staticmethod
    def Omega(u, p):
        return np.eye(1)

    @staticmethod
    def invOmega(u, p):
        return np.eye(1)

    @staticmethod
    def lamb(u, p):
        return [u]

    @staticmethod
    def aMax(u, p):
        return abs(u)

    @staticmethod
    def R(uL, uR, pL, pR, bctype):
        if bctype == Problem.LEFTBC:
            return (None, Hopf.F(np.array([0.5]), pR))
        if bctype == Problem.RIGHTBC:
            return (Hopf.F(uL, pL), None)
        if uL[0] >= uR[0]:
            D = 0.5 * (uL[0] + uR[0])
            if D >= 0:
                return (Hopf.F(uL, pL), Hopf.F(uL, pL))
            else:
                return (Hopf.F(uR, pR), Hopf.F(uR, pR))
        else:
            if uL[0] > 0:
                return (Hopf.F(uL, pL), Hopf.F(uL, pL))
            elif uR[0] < 0:
                return (Hopf.F(uR, pR), Hopf.F(uR, pR))
            else:
                z = np.zeros(1)
                return (Hopf.F(z, None), Hopf.F(z, None))
