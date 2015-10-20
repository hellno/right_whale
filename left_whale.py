import matplotlib.pyplot as plt

from skimage import data, feature
from skimage.filters import threshold_otsu, threshold_adaptive
import scipy as sp
import numpy as np

# def find_blob_corners(blob, width, height):
#     """
#     finds coordinates for the four corners of a given blob
#     blob is in (y,x)
#     returns array with four tuples in (x,y):
#     lower_left, upper_left, upper_right, lower_right  
#     """
#     down = 0, left = 0, up = height, right = width

#     for b in blob:
#         if(b[1] < left):
#             left = b[1]
#         if(b[1] > right):
#             right = b[1]
#         if(b[0] < down):
#             down = b[0]
#         if(b[0] > up):
#             up = b[0]

#     return [(left,down), (left,up), (right, up), (right, down)]

image = sp.misc.imread('w_26.jpg', flatten=True)

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 100
binary_adaptive = threshold_adaptive(image, block_size, offset=10)

# blob = feature.blob_dog(binary_adaptive, threshold=.5, max_sigma=40)
# corners = find_blob_corners(blob, image.shape[0])

fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
ax0, ax1 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_adaptive)
ax1.set_title('Adaptive thresholding')

# ax1.contour(blob[:, 1], blob[:, 0], linewidth=2, color='y')

plt.show()