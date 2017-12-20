from dg import Dg
from advection import Advection
from wave import Wave

def main():
#    prob = Advection()
    prob = Wave()
    dg = Dg(50, 4, prob, C=0.05, use_limiter=True)
    dg.run()

if __name__ == "__main__":
    main()
