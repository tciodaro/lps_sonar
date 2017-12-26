import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.neighbors import KernelDensity
import numpy as np

plt.close('all')

# Classes
data = ds.make_blobs(n_samples=1000, n_features=1, centers=1, cluster_std=.8)


# Estimate kernel density
density = KernelDensity(kernel='gaussian', bandwidth=0.6)
density.fit(data[0])

# Evaluate domain
X = np.linspace(np.min(data[0][:,0]), np.max(data[0][:,0]), 100)
X = np.array([X]).transpose()

Z =  np.exp(density.score_samples(X))


# Plot the data and the density
plt.plot(data[0][:,0], np.zeros(data[0].shape[0]), 'ok', markeredgecolor='none')
plt.plot(X, Z,'-r', lw=3)

plt.ylim([-0.001, Z.max()*1.2])



# Plot some levels
for ax in np.linspace(X.min(), X.max(), 10):
    az = np.exp(density.score_samples(ax))
    plt.plot([X.min(), X.max()],[az, az], '--y',lw=2)
    #plt.plot([ax,ax],[0, az], '--y',lw=2)

# Plot the desired level

zthrs = np.linspace(Z.min(), Z.max(), 100)
zsum = np.array([ [(Z >= th).sum() / float(Z.size), th] for th in zthrs])

iarg = np.nonzero(zsum[:,0] >= 0.9)[0][-1]
plt.plot([X.min(), X.max()], [zthrs[iarg], zthrs[iarg]], '--ko',lw=2)

#raise Exception('STOP')

