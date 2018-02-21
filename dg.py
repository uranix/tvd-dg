import numpy as np
from matrix import Matrices
from problem import Problem

class Dg(object):
    """
    Construct a discontinunous Galerkin solver

    Parameters:
    n    -- number of segments
    p    -- polynomial degree
    prob -- an instance of Problem to solve
    """
    def __init__(self, n, p, prob, C=0.5, method='tvd'):
        # Store parameters
        self.prob = prob
        self.n = n
        self.p = p
        self.h = prob.L() / n
        self.C = C
        self.method = method

        self.mat = Matrices(p)

    """
    Solve the problem
    """
    def run(self):
        prob = self.prob
        n = self.n
        p = self.p
        h = self.h
        mat = self.mat

        nvars = len(prob.u0(0))

        u = np.empty((n, p + 1, nvars), dtype=np.double)
        params = []
        self.params = params

        for i in range(n):
            for j in range(p+1):
                x = (i + mat.s[j] / 2 + 0.5) * h
                u[i, j] = prob.u0(x)
            params.append(prob.p(x))

        t = 0
        stepnum = 0
        self.save_csv(stepnum, u)
        while t < prob.T():
            amax = 0
            for i in range(n):
                for j in range(p+1):
                    a = prob.aMax(u[i, j], params[i])
                    if a > amax:
                        amax = a

            tau = self.C * h / amax;
            u = self.step(u, tau, t)
            t += tau
            stepnum += 1

            if (stepnum % 5 == 0):
                print("t = %8.4f" % t)
                self.save_csv(stepnum, u)

    """
    Perform a single timestep using a third order
    TVD Runge-Kutta integrator
    """
    def step(self, u, tau, t):
        Au = self.A(u, t, tau)
        umid = u + Au * tau

        Au = self.A(umid, t + tau, tau)
        umid = 0.75 * u + 0.25 * (umid + tau * Au)

        Au = self.A(umid, t + tau / 2, tau);
        u = (1.0 / 3) * u + (2.0 / 3) * (umid + tau * Au)
        return u

    """
    Evaluate solution u at segment i in point s.
    s = -1 corresponds to the left edge of the segment and
    s = 1 corresponds to the right
    """
    def interp(self, u, i, s):
        L = self.mat.get_interpolate_vector(s)
        return L @ u[i, :, :]

    def flux_HO(self, u):
        n = self.n
        p = self.p
        prob = self.prob
        params = self.params

        FL = np.empty((n, u.shape[2]))
        FR = np.empty((n, u.shape[2]))

        for i in range(n+1):
            uL = self.interp(u, i-1, 1) if i > 0 else None
            uR = self.interp(u, i,  -1) if i < n else None
            pL = params[i-1] if i > 0 else None
            pR = params[i]   if i < n else None
            bct = Problem.INNER
            if i == 0:
                bct = Problem.LEFTBC
            if i == n:
                bct = Problem.RIGHTBC

            FRv, FLv = prob.R(uL, uR, pL, pR, bct)
            if i > 0:
                FR[i-1] = FRv
            if i < n:
                FL[i]   = FLv

        Fui = np.empty((p+3, u.shape[2]))
        FHO = np.empty((n, p+2, u.shape[2]))
        Ki = self.mat.get_Ki()

        for i in range(n):
            for j in range(p+1):
                Fui[j + 1] = prob.F(u[i, j], params[i])
            Fui[0], Fui[p+2] = FL[i], FR[i]

            KF = Ki @ Fui

            FHO[i, 0] = Fui[0]
            for j in range(0, p+1):
                FHO[i, j+1] = FHO[i, j] - KF[j]

        return FHO

    def flux_LO(self, u):
        n = self.n
        p = self.p
        prob = self.prob
        params = self.params

        FL = np.empty((n, u.shape[2]))
        FR = np.empty((n, u.shape[2]))

        for i in range(n+1):
            uL = u[i-1, p] if i > 0 else None
            uR = u[i, 0]   if i < n else None
            pL = params[i-1] if i > 0 else None
            pR = params[i]   if i < n else None
            bct = Problem.INNER
            if i == 0:
                bct = Problem.LEFTBC
            if i == n:
                bct = Problem.RIGHTBC

            FRv, FLv = prob.R(uL, uR, pL, pR, bct)
            if i > 0:
                FR[i-1] = FRv
            if i < n:
                FL[i]   = FLv

        FLO = np.empty((n, p+2, u.shape[2]))

        for i in range(n):
            FLO[i, 0] = FL[i]
            FLO[i, p+1] = FR[i]

            pp = params[i]

            for j in range(1, p+1):
                FLO[i, j], _ = prob.R(u[i, j-1], u[i, j], pp, pp, Problem.INNER)

        return FLO

    """
    Evaluate the right hand side for the DG ODE
    """
    def A(self, u, t, dt):
        n = self.n
        p = self.p
        prob = self.prob
        params = self.params
        method = self.method

        if method == 'tvd' or method == 'ho':
            FHO = self.flux_HO(u)
        if method == 'tvd' or method == 'lo':
            FLO = self.flux_LO(u)

        if method == 'lo':
            FLUX = FLO

        if method == 'ho':
            FLUX = FHO

        if method == 'tvd':
            delta = np.zeros_like(FHO)
            omega = np.zeros((n, p+2, delta.shape[2], delta.shape[2]))
            invomega = np.zeros((n, p+2, delta.shape[2], delta.shape[2]))
            lamb = np.zeros_like(delta)

            for i in range(n):
                for j in range(1, p+1):
                    uavg = 0.5 * (u[i, j, :] + u[i, j-1, :])
                    omega[i, j, :, :]    = prob.Omega(uavg, params[i])
                    invomega[i, j, :, :] = prob.invOmega(uavg, params[i])
                    lamb[i, j, :]        = prob.lamb(uavg, params[i])
                    delta[i, j, :] = omega[i, j, :, :] @ (u[i, j, :] - u[i, j-1, :])
                if i > 0:
                    uavg = 0.5 * (u[i, 0, :] + u[i-1, p, :])
                    omega[i, 0, :, :]    = prob.Omega(uavg, params[i])
                    invomega[i, 0, :, :] = prob.invOmega(uavg, params[i])
                    lamb[i, 0, :]        = prob.lamb(uavg, params[i])
                    delta[i, 0, :] = omega[i, 0, :, :] @ (u[i, 0, :] - u[i-1, p, :])
                if i < n-1:
                    uavg = 0.5 * (u[i+1, 0, :] + u[i, p, :])
                    omega[i, p+1, :, :]    = prob.Omega(uavg, params[i])
                    invomega[i, p+1, :, :] = prob.invOmega(uavg, params[i])
                    lamb[i, p+1, :]        = prob.lamb(uavg, params[i])
                    delta[i, p+1, :] = omega[i, p+1, :, :] @ (u[i+1, 0, :] - u[i, p, :])

            ideal = np.zeros_like(FHO)
            for i in range(n):
                for j in range(p+2):
                    ideal[i, j] = omega[i, j, :, :] @ (FHO[i, j, :] - FLO[i, j, :])

            delta = delta.reshape(n * (p+2), -1)
            ideal = ideal.reshape(n * (p+2), -1)
            lamb  = lamb.reshape(n * (p+2), -1)

            def minmod(x, y):
                sx = np.sign(x)
                sy = np.sign(y)
                return 0.5 * (sx + sy) * np.minimum(abs(x), abs(y))

            for ij in range(1, n * (p+2) - 1):
                lamplus = lamb[ij, :]
                lamminus = lamplus.copy()
                lamplus[lamplus < 0] = 0
                lamminus[lamminus > 0] = 0
                sleft  = minmod(delta[ij], delta[ij-1])
                sright = minmod(delta[ij], delta[ij+1])
                val = np.diag(lamplus) @ sleft - np.diag(lamminus) @ sright
                ideal[ij] = minmod(ideal[ij], val)

            ideal = ideal.reshape(n, p+2, -1)

            FLUX = np.zeros_like(FHO)

            for i in range(n):
                for j in range(p+2):
                    FLUX[i, j] = FLO[i, j] + invomega[i, j, :, :] @ ideal[i, j]

        Au = np.empty_like(u)

        for i in range(n):
            for l in range(p+1):
                Au[i, l] = (FLUX[i, l] - FLUX[i, l+1]) * 2 / (self.h * self.mat.weight[l])

        return Au

    """
    Save current solution to CSV file
    """
    def save_csv(self, stepnum, u):
        n = self.n
        filename = '_'.join([
                type(self.prob).__name__,
                self.method,
                str(stepnum) + '.csv'
            ])
        with open('csv/' + filename, 'w') as csv:
            csv.write('x')
            for j in range(u.shape[2]):
                csv.write(',u%d' % j)
            csv.write('\n')
            for i in range(n):
                for j in range(u.shape[1]):
                    ui = u[i, j]
                    s = self.mat.s[j]
                    csv.write('%e' % ((i + 0.5*s + 0.5)*self.h))
                    for uiv in ui:
                        csv.write(',%e' % uiv)
                    csv.write('\n')
                """
                for s in np.linspace(-1, 1, 21):
                    ui = self.interp(u, i, s);
                    csv.write('%e' % ((i + 0.5*s + 0.5)*self.h))
                    for uiv in ui:
                        csv.write(',%e' % uiv)
                    csv.write('\n')
                """

    """
    Save current solution to DAT file
    """
    def save_dat(self, stepnum, u):
        n = self.n
        filename = '_'.join([
                type(self.prob).__name__,
                self.method,
                str(stepnum) + '.dat'
            ])

        logrid = [-1]
        for w in self.mat.weight:
            logrid.append(logrid[-1] + w)

        with open('csv/' + filename, 'w') as csv:
            csv.write('\n')
            for i in range(n):
                for s in np.linspace(-1, 1, 21):
                    ui = self.interp(u, i, s);
                    csv.write('%e' % ((i + 0.5*s + 0.5)*self.h))
                    for uiv in ui:
                        csv.write(' %e' % uiv)
                    csv.write('\n')
                csv.write('\n')
