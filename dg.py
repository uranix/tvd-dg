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
    def __init__(self, n, p, prob, C=0.5, use_limiter=False):
        # Store parameters
        self.prob = prob
        self.n = n
        self.p = p
        self.h = prob.L() / n
        self.C = C
        self.use_limiter = use_limiter

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
        self.save(stepnum, u)
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
                self.save(stepnum, u)

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

    """
    Evaluate total variation for a scalar-valued function
    """
    def TV(self, u):
        return np.sum(np.abs(np.diff(u)))

    """
    Evaluate the right hand side for the DG ODE
    """
    def A(self, u, t, dt):
        n = self.n
        p = self.p
        prob = self.prob
        params = self.params

        Au = np.empty_like(u)

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

        omega = np.empty((n, u.shape[2], u.shape[2]))
        inv_omega = np.empty_like(omega)
        S = np.empty_like(u)
        for i in range(n):
            Uiavg = self.mat.weight @ u[i, :]
            omega[i] = prob.Omega(Uiavg, params[i])
            inv_omega[i] = np.linalg.inv(omega[i])
            for j in range(p+1):
                S[i, j] = omega[i] @ u[i, j]

        Fui = np.empty((p+3, u.shape[2]))

        for i in range(n):
            for j in range(p+1):
                Fui[j + 1] = prob.F(u[i, j], params[i])
            Fui[0], Fui[p+2] = FL[i], FR[i]

            Ki = self.mat.get_Ki()
            KF = Ki @ Fui

            for l in range(p+1):
                Au[i, l] = KF[l] * 2 / (self.h * self.mat.weight[l])

        if not self.use_limiter:
            return Au

        unew = u + dt * Au
        Snew = np.empty_like(unew)
        for i in range(n):
            for j in range(p+1):
                Snew[i, j] = omega[i] @ unew[i, j]

        deltaS = np.empty_like(Snew)

        for k in range(u.shape[2]):
            # Treat each invariant separately
            s    = S   [:, :, k].flatten()
            snew = Snew[:, :, k].flatten()

            TVold = self.TV(s)
            TVnew = self.TV(snew)
            print(TVold, TVnew)

        deltaS[:, :, :] = 0

        for i in range(n):
            for j in range(p+1):
                Au[i, j] += inv_omega[i] @ deltaS[i, j]

        return Au

    """
    Save current solution to CSV file
    """
    def save(self, stepnum, u):
        n = self.n
        filename = '_'.join([
                type(self.prob).__name__,
                ('tvd' if self.use_limiter else 'notvd'),
                str(stepnum) + '.csv'
            ])
        with open('csv/' + filename, 'w') as csv:
            csv.write('x')
            for j in range(u.shape[2]):
                csv.write(',u%d' % j)
            csv.write('\n')
            for i in range(n):
                for s in np.linspace(-1, 1, 21):
                    ui = self.interp(u, i, s);
                    csv.write('%e' % ((i + 0.5*s + 0.5)*self.h))
                    for uiv in ui:
                        csv.write(',%e' % uiv)
                    csv.write('\n')
