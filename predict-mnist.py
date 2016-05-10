import ast
import itertools
from scipy import linalg
import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


#2d-data
with open("mnist-64.txt", "r") as f:
  l = ast.literal_eval(f.readline())
  clusters = ast.literal_eval(f.readline())
X = np.array(l)
dpgmm = mixture.DPGMM(n_components=10, alpha = 2, covariance_type='full')
dpgmm.fit(X)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])

for i, (clf, title) in enumerate([(dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-10, 10)
    plt.ylim(-5, 10)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()
