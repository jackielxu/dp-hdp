import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.animation as animation
import random
import scipy.stats as stats
import scipy.linalg
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import ast

def load_data(filename):
  """
  Loads data and the correct clusters from a file.
  
  Input:
  filename : string with the filename

  Output:
  data : the data points as a list of points ([x,y])
  clusters : the number cluster that each point is in; same order as the list of points
  """
  with open(filename, "r") as f:
    data = ast.literal_eval(f.readline()) 
    clusters = ast.literal_eval(f.readline())
  return data, clusters


def load_mnist(n_components):
  """
  Loads MNIST data as vectors. 
  
  Input:
  n_components : the number of components to reduce to for PCA

  Output:
  data : MNIST data
  """
  pca = PCA(n_components=n_components)
  digits = load_digits()
  data = scale(digits.data)
  data_reduced = pca.fit_transform(data)
  labels = digits.target
  data_reduced_list = [list(i) for i in list(data_reduced)]
  labels_list = list(labels)
  with open("mnist-{}.txt".format(n_components), "w") as f:
    f.write(str(data_reduced_list))
    f.write("\n")
    f.write(str(labels_list))
  return data_reduced_list, labels_list

def sample_niw(mu_0, lambda_0, kappa_0, nu_0):
  lmbda = sample_invishart(lmbda_0,nu_0) # lmbda = np.linalg.inv(sample_ishart(np.linalg.inv(lmbda_0),nu_0))
  mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)
  return mu, lmbda 


def sample_invwishart(lmbda,dof):
  n = lmbda.shape[0]
  chol = np.linalg.cholesky(lmbda) 
  if (dof <= 81+n) and (dof == np.round(dof)):
    x = np.random.randn(dof,n)
  else:
    x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(n)))))
    x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2) 
  R = np.linalg.qr(x,'r')
  T = scipy.linalg.solve_triangular(R.T,chol.T).T
  return np.dot(T,T.T) 

def update_map(mu_0, lambda_0, kappa_0, nu_0, n, x_bar, Psi_0, Sigma_0, x_mean, x_i):
    mu_n = ((kappa_0 * mu_0) + n * x_bar)/(kappa_0 + n)
    kappa_n = kappa_0 + n
    nu_n = nu_0 + n
    C = sum([np.dot((x_i[i] - x_mean), (x_i[i] - x_mean)) for i in range(len(x_i))])
    Psi_n = Psi_0 + C + (kappa_0 * n)/(kappa_0 + n)*np.dot((x_bar - mu_0), (x_bar - mu_0))
    Sigma_n = (kappa_n + 1)/(kappa_n*(nu_n - len(x_mean) + 1))*Psi_n 
    return mu_n, kappa_n, nu_n, Psi_n, Sigma_n

def gibbs(x, c_guess, a_0, t, F):
  """
  x_i are randomly assigned cluster assignments z_i.
  
  At each iteration, these cluster assignments are updated? How exactly is this done?  
  We look at an x_i, we sample it from some posterior distribution, but where do we get this from?  
  """
  old_z = np.random.randint(1, 2, len(x)) # Random cluster assignments
  old_z = [1]*len(x)
  new_z = []
  
  for iter_i in range(t):
    # u = pu(1, np.random.normal(0, 1, len(x)), len(x))
    for i, x_i in enumerate(x):
      rand = random.random()
      if rand < (a_0 / float(a_0 + i)):
        q_0 = 1.0/(2.0 * np.sqrt(np.pi)) * np.exp(-(np.square(x_i))/4.0)
        H = np.random.normal(np.array(x_i)/2.0, 0.5)
        temp_z = round(a_0 * q_0 * H)
        new_z = new_z + [temp_z]
      else:
        # F = lambda u : map(int, map(round, np.random.normal(u, 1)*6)) # 1D data
        temp_z = F(random.choice(old_z[:i]))
        new_z = new_z + [temp_z]
        # z[i] = random.choice(F(u)) 
    old_z = new_z[:]
    new_z = []

  return old_z
 
def dpmm(G_0, F, alpha, n, dim=1):
  """
  Generates data via a DPMM. Currently, plots 1D or 2D data. 

  Input
    G_0 : base distribution
      F : generative distribution
  alpha : concentration parameter
      n : number of data points to generate

  Output:
  clusters : mapping from cluster numbers to the points in the cluster
  """
  clusters = {} # Maps cluster numbers to the points in the cluster
  
  z = crp(alpha, n) # Generate cluster assignments for each point
  
  # Match cluster assignments to params
  for z_i in z:
    theta_i = G_0[z_i]

    if dim == 1:
      X = F(theta_i)
    else:
      X = list(F(theta_i)) # Can be multivariate

    # Generate data points
    if z_i in clusters:
      clusters[z_i].append(X)
    else:
      clusters[z_i] = [X]

  return clusters


def crp(alpha, n=200):
  """
  Generates data via Chinese Restaurant Process. 
  
  Input
  alpha : concentration parameter
      n : number of data points

  Output
  out : output distribution
  """
  out = [1]
  count = 1
  for i in range(1,n+1):
    p_new = alpha / float(i + alpha)
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


def pu(alpha, H, n=200):
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
  return int(round(next(H)*5)) 

## MNIST data
'''
load_mnist(2)
load_mnist(5)
load_mnist(10)
load_mnist(20)
load_mnist(30)
load_mnist(50)
load_mnist(64)
'''

## Testing the Gibbs Sample
"""
with open("1d-data.txt", "r") as f:
  l = ast.literal_eval(f.readline())
  true_clusters = ast.literal_eval(f.readline())
x = l
c_guess = 3
a_0 = 2
t = 100
# F = lambda u : map(int,map(round,np.random.normal(u, 0.25))) 
F = lambda u : int(round(np.random.normal(u, 1)))
for i in range(10):
  z = gibbs(x, c_guess, a_0, t, F)
  z_min = min(z)
  z = [i + abs(z_min) + 1 for i in z] # TODO: This is a temporary fix; the values for the cluster assignments are too high to plot (matplotlib doesn't provide that many colors)
  print z
  print true_clusters
  plt.scatter(x, [0]*len(x), c=z)
  plt.show()
"""
## Testing the DPMM
# Generating 1D data
"""
alpha = 2
n = 100
G_0 = np.random.uniform(0, 10, n)
F = lambda u : np.random.normal(u, 0.25) # 1D data
clusters = dpmm(G_0, F, alpha, n)

# Associate points with colors
points = []
colors = []
for i in clusters:
  for val in clusters[i]:
    points.append(val)
    colors.append(i)

with open("1d-data.txt", "w") as f:
  f.write(str(points)) 
  f.write("\n")
  f.write(str(colors))

plt.scatter(points, [0]*len(points), c=colors)
plt.show()
"""
# Generating 2D data
"""
alpha = 2
n = 500
G_01 = np.random.uniform(0, 10, n)
G_02 = np.random.uniform(0, 10, n)
G_0 = zip(G_01, G_02) 

F = lambda u: np.random.multivariate_normal(u, np.array([[1,0],[0,1]])*0.25) # 2D data
clusters = dpmm(G_0, F, alpha, n, dim=2)

# Associate points with colors
points = []
colors = []
for i in clusters:
  for val in clusters[i]:
    points.append(val)
    colors.append(i)

with open("2d-data.txt", "w") as f:
  f.write(str(points)) 
  f.write("\n")
  f.write(str(colors))

x = []
y = []
for i in points:
  x.append(i[0])
  y.append(i[1])

plt.scatter(x,y, c=colors)
plt.show()
"""

"""
## Testing the CRP
crp_out = crp(1)
print(crp_out)
plt.hist(crp_out)
plt.show()

## Testing the Polya-Urn
pu_out = pu(1, np.random.normal(10, 5, 1000))
plt.hist(pu_out)
plt.show()

## Testing the Stick-Breaking Process
sbp_out = sbp(1, 10)
plt.bar([i for i in range(10)], sbp_out)
plt.show()
"""
