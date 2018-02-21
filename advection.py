import math
import numpy as np
from problem import Problem

from param import Param

class Advection(Problem):
    @staticmethod
    def u0(x):
        return [0]

    @staticmethod
    def p(x):
        return Param(1, 1)

    @staticmethod
    def c(p):
        return math.sqrt(p.mu / p.rho)

    @staticmethod
    def F(u, p):
        return Advection.c(p) * u

    @staticmethod
    def T():
        return 0.8

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
        return [Advection.c(p)]

    @staticmethod
    def aMax(u, p):
        return Advection.c(p)

    @staticmethod
    def R(uL, uR, pL, pR, bctype):
        if bctype == Problem.LEFTBC:
            return (None, Advection.F(np.array([1]), pR))
        if bctype == Problem.RIGHTBC:
            return (Advection.F(uL, pL), None)
        return (Advection.F(uL, pL), Advection.F(uL, pL))
