
class Problem(object):
    INNER   = 0
    LEFTBC  = 1
    RIGHTBC = 2

    """
    Termination time
    """
    @staticmethod
    def T():
        raise NotImplementedError

    """
    Domain size
    """
    @staticmethod
    def L():
        raise NotImplementedError

    """
    A fuction returning the initial state at point x
    """
    @staticmethod
    def u0(x):
        raise NotImplementedError

    """
    A fuction returning the medium parameters at point x
    """
    @staticmethod
    def p(x):
        raise NotImplementedError

    """
    Differential problem flux. The equation is
    :math:`u_t + F(u)_x = 0`
    F may also depend on medium parameters p 
    """
    @staticmethod
    def F(u, p):
        raise NotImplementedError

    """
    Returns the left eigevectors matrix
    """
    @staticmethod
    def Omega(u, p):
        raise NotImplementedError

    """
    Returns the inverse of the left eigevectors matrix
    """
    @staticmethod
    def invOmega(u, p):
        raise NotImplementedError

    """
    Returns the eigenvalues as a vector
    """
    @staticmethod
    def lamb(u, p):
        raise NotImplementedError

    """
    Returns the fastest travelling wave speed
    """
    @staticmethod
    def aMax(u, p):
        raise NotImplementedError

    """
    Solves the Riemann problem
    """
    @staticmethod
    def R(uL, uR, pL, pR, bctype):
        raise NotImplementedError
