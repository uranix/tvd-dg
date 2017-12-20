import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange

class Matrices(object):
    def __init__(self, p):
        self.s, self.weight = leggauss(p+1)
        self._int_mat = {}
        Ki = np.empty((p+1, p+3))
        Ki[:, 0]   = self.get_interpolate_vector(-1)
        Ki[:, p+2] = -self.get_interpolate_vector(1)
        I = np.eye(p+1)
        for i in range(p+1):
            dellds = lagrange(self.s, I[i]).deriv()
            for j in range(p+1):
                Ki[i, j+1] = self.weight[j] * dellds(self.s[j])
        self._Ki = Ki
    def get_interpolate_vector(self, z):
        _int_mat = self._int_mat
        if z in _int_mat:
            return _int_mat[z]
        I = np.eye(len(self.s))
        Z = np.empty(I.shape[0])
        for i in range(len(self.s)):
            Z[i] = lagrange(self.s, I[i])(z)
        _int_mat[z] = Z
        return Z
    def get_Ki(self):
        return self._Ki
