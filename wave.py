import math
import numpy as np
from problem import Problem

from param import Param

class Wave(Problem):
    @staticmethod
    def u0(x):
        if x > 0.2 and x < 0.4:
            return [1, -1]
        return [0, 0]

    @staticmethod
    def p(x):
        if x < 0.7:
            return Param(1, 1)
        return Param(2, 2)

    @staticmethod
    def a(p):
        return math.sqrt(p.mu * p.rho)

    @staticmethod
    def F(u, p):
        return [-u[1] / p.rho, -u[0] * p.mu]

    @staticmethod
    def T():
        return 0.4

    @staticmethod
    def L():
        return 1.0

    @staticmethod
    def Omega(u, p):
        a = Wave.a(p)
        return [[a, 1], [-a, 1]]

    @staticmethod
    def aMax(u, p):
        return math.sqrt(p.mu / p.rho)

    @staticmethod
    def R(uL, uR, pL, pR, bctype):
        if bctype == Problem.LEFTBC:
            a = Wave.a(pR);
            uM = [0, uR[1] + uR[0] * a]
            return (None, Wave.F(uM, pR))
        if bctype == Problem.RIGHTBC:
            a = Wave.a(pL);
            uM = [0, uL[1] - uL[0] * a]
            return (Wave.F(uM, pL), None)
        if bctype == Problem.INNER:
            ul, sl = uL
            ur, sr = uR
            mul = pL.mu
            mur = pR.mu
            rol = pL.rho
            ror = pR.rho

            al = Wave.a(pL);
            ar = Wave.a(pR);

            if mul == mur and rol == ror:
                nu = 1e20
            else:
                nu = 10

            gamma = 1 / (al * ar + nu * (al + ar))
            ulstar = gamma * (-(ar*sl) - nu*sl + nu*sr + al*ar*ul + al*nu*ul + ar*nu*ur)
            vlstar = gamma * (ar*nu*sl + al*nu*sr - al*ar*nu*ul + al*ar*nu*ur)
            urstar = gamma * (-(nu*sl) + al*sr + nu*sr + al*nu*ul + al*ar*ur + ar*nu*ur)
            vrstar = gamma * (ar*nu*sl + al*nu*sr - al*ar*nu*ul + al*ar*nu*ur)
            ULstar = [ulstar, vlstar]
            URstar = [urstar, vrstar]

            return (Wave.F(ULstar, pL), Wave.F(URstar, pR))
        return None
