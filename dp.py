import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.animation as animation
import random

# TODO: Write a DP (Chinese Restaurant Process)

def crp(alpha, n=200):
  """
  Generates data via Chinese Restaurant Process. 
  
  Input
  a : concentration parameter
  n : number of data points

  Output
  out : output distribution
  """
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

def sbp(alpha, n=200):
  """
  Generative process that simulates the stick breaking process.

  Input
  alpha : concentration/disperson parameter; roughly correlates to how spread
          out we want our diversity of stick lengths to be.  
      n : number of iterations to continue the stick breaking 

  Output
  out : output distribution
  """

  out = [] 
  draws = iter(np.random.beta(1, alpha, n))
  pi_k = 1
  stk = 1

  for i in range(n):
    b_k = next(draws)
    pi_k = b_k * stk
    out.append(pi_k) 
    stk = stk * (1 - b_k)

  return out


def pu(alpha, H, n=100):
  """
  Generative process that simulates a Polya Urn scheme. We index the colors
  in both the H and G urns by integers. 

  Input
  alpha:  concentration/dispersion parameter; roughly correlates to how spread
          out we want our diversity of colors to be. 

      H:  base distribution; specifies a distribution for an underlying color
          scheme that is sampled at a rate proportional to alpha. 

          *NOTE*: distribution must be one from np.random, such as 
          np.random.normal, or np.random.beta, and the size parameter must 
          match n.

      n:  number of iterations to draw samples from an urn in one run.

  Output
  out : output distribution
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

crp_out = crp(10)
#plt.hist(out)

pu_out = pu(10.0, np.random.normal(10, 5, 1000), 1000)
# plt.hist(pu_out)

sbp_out = sbp(1, 10)
plt.bar([i for i in range(10)], sbp_out)
plt.show()
