import numpy as np
import matplotlib.pyplot as plt

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


#TODO: Write a DP (Polya Urn)




