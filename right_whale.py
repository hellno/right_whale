import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

def flatten_image(img):
    """
    takes an (m, n) numpy array and flattens it to (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s, 4)

    return img_wide[0]

whale = sp.misc.imread('test_whale_square.png', flatten=True)

# plt.imshow(whale)
# plt.show()
print whale.shape

# downsize by a factor of 4
whale = whale[::2, ::2] + whale[1::2, ::2] + whale[::2, 1::2] + whale[1::2, 1::2]
whale = whale[::2, ::2] + whale[1::2, ::2] + whale[::2, 1::2] + whale[1::2, 1::2]

graph = image.img_to_graph(whale)
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / whale.std()) # + eps
N_REGIONS = 11
print 'loop go'

# for assign_labels in ('kmeans','discretize'):
t0 = time.time()
print 't0 %.2fs' % t0
labels = spectral_clustering(graph, n_clusters = N_REGIONS,
                             assign_labels='discretize', 
                             random_state=1, 
                             eigen_solver='arpack')

t1 = time.time()
print 't1 %.2fs' % t1
labels = labels.reshape(whale.shape)

plt.figure(figsize=(5,5))
plt.imshow(whale, cmap=plt.cm.gray)

print 'looploop gogo'

for l in range(N_REGIONS):
    plt.contour(labels == l, contours=1,
                colors=[plt.cm.spectral(l /float(N_REGIONS)), ])

plt.xticks(())
plt.yticks(())

print 'Spectral clustering: %s, %.2fs' % ('discretize', (t1 - t0))

plt.show()