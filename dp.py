import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.animation as animation
import random

# TODO: Write a DP (Chinese Restaurant Process)

# Test the DP
a = [1,1]
G = np.random.dirichlet(a)
#plt.plot(G)

def crp(a, N):
  for n in N:
    p_new = a / float(n + a)
    p_old = table_size / float(n + a)


#TODO: Write a DP (Stick Breaking Process)

def sbp(alpha, H, n=200):
    for i in range(100):
        print(np.random.beta(2, 3))
    pass

#TODO: Write a DP (Polya Urn)
def pu(alpha, H, n=100):
    """
    Generative process that simulates a Polya Urn scheme. We index the colors
    in both the H and G urns by integers. 

    alpha:  concentration/dispersion parameter; roughly correlates to how spread
            out we want our diversity of colors to be. 

        H:  base distribution; specifies a distribution for an underlying color
            scheme that is sampled at a rate proportional to alpha. 

            *NOTE*: distribution must be one from np.random, such as 
            np.random.normal, or np.random.beta, and the size parameter must 
            match n.

        n:  number of iterations to draw samples from an urn in one run.

    """
    
    H_draws = iter(H)
    out = [sample(H_draws)]
    alpha = float(alpha)

    for i in range(n):
        rand = random.random()
        if rand < alpha/(alpha + i):
            nxt = sample(H_draws)
            out.append(nxt)
        else:
            out.append(random.choice(out))

    return out

def sample(H):
    return int(max(round(next(H)), 0))

out = pu(10.0, np.random.normal(10, 5, 100), 100)
plt.hist(out)
plt.show()
