import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.animation as animation
import random

# TODO: Write a DP (Chinese Restaurant Process)

def crp(a, n=200):
  out = [1]
  count = 1
  for i in range(1,n+1):
    p_new = a / float(i + a)
    rand = random.random()
    if rand <= p_new:
      count += 1
      out.append(count)
    else:
      choice = random.choice(out)
      out.append(choice)
  return out

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
    return int(max(round(next(H)), 0)) #TODO: Should this be max of the number and 0?

#out = pu(10.0, np.random.normal(10, 5, 100), 100)
out = crp(10)
plt.hist(out)
plt.show()
