import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.neighbors import KernelDensity
import numpy as np

# Classes
data = ds.make_blobs(n_samples=1000, n_features=2, centers=[[0,0],[3,2],[5,2],[0,1]], cluster_std=.5)


# Estimate kernel density
density = KernelDensity(kernel='gaussian', bandwidth=0.6)
density.fit(data[0])

# Evaluate domain
randomsamples = density.sample(10000)
Zsim =  np.exp(density.score_samples(randomsamples))
Zreal = np.exp(density.score_samples(data[0]))


X = np.linspace(randomsamples[:,0].min(), randomsamples[:,0].max(), 50)
Y = np.linspace(randomsamples[:,1].min(), randomsamples[:,1].max(), 50)
xds, yds = np.meshgrid(X,Y)
pos = np.vstack((xds.ravel(), yds.ravel())).T
zds = np.reshape(np.exp(density.score_samples(pos)).T, xds.shape)

xavg = randomsamples[:,0].mean()
yavg = randomsamples[:,1].mean()
xstd = randomsamples[:,0].std()
ystd = randomsamples[:,1].std()

# Plot
plt.plot(randomsamples[:,0], randomsamples[:,1], '.k', markeredgecolor='none')
plt.plot(data[0][:,0], data[0][:,1], '.r', markeredgecolor='none')
h_c = plt.contour(xds, yds, zds, 10, colors='y')
#plt.plot(xds, yds, 'or', markeredgecolor='none')

#plt.plot([xavg+xstd, xavg+xstd], [Y.min(), Y.max()], '--y',lw=3)
#plt.plot([xavg+xstd*2, xavg+xstd*2], [Y.min(), Y.max()], '--y',lw=3)
#plt.plot([X.min(), X.max()], [yavg+ystd, yavg+ystd],  '--y',lw=3)
#plt.plot([X.min(), X.max()], [yavg+ystd*2, yavg+ystd*2], '--y',lw=3)
#raise Exception('STOP')


Zreal.sort()
zthrs = np.linspace(Zreal.min(), Zreal.max(), 10000)
zsum = np.array([ [(Zreal >= th).sum() / float(Zreal.size), th] for th in zthrs])

iarg = np.argmax(zsum[zsum[:,0] >= 0.9, 1])
zvalue = Zreal[np.nonzero(Zreal >= zsum[iarg,1])][0]
plt.contour(xds, yds, zds, levels=[zvalue], colors='r', linewidths=[3])



Zsim.sort()
zthrs = np.linspace(Zsim.min(), Zsim.max(), 10000)
zsum = np.array([ [(Zsim >= th).sum() / float(Zsim.size), th] for th in zthrs])

iarg = np.argmax(zsum[zsum[:,0] >= 0.9, 1])
zvalue = Zsim[np.nonzero(Zsim >= zsum[iarg,1])][0]
plt.contour(xds, yds, zds, levels=[zvalue], colors='b', linewidths=[3])
