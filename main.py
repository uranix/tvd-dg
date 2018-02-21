from dg import Dg
from advection import Advection
from wave import Wave
from hopf import Hopf

def main():
#    prob = Advection()
    prob = Wave()
#    prob = Hopf()

    print('\nLO\n')
    dg = Dg(50, 4, prob, C=0.05, method='lo')
    dg.run()
    print('\nHO\n')
    dg = Dg(50, 4, prob, C=0.05, method='ho')
    dg.run()
    print('\nTVD\n')
    dg = Dg(50, 4, prob, C=0.05, method='tvd')
    dg.run()

if __name__ == "__main__":
    main()
