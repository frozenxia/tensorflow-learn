import matplotlib.pyplot as plt
import pylab
import cv2

import numpy as np

img = plt.imread('1.jpeg')
plt.figure(figsize=(13, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)

fil = np.array([[4, 0, 0, 0, 0],
                [0, -1, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, -1, 0],
                [0, 0, 0, 0, -1]])

res = cv2.filter2D(img, -1, fil)
plt.subplot(1, 2, 2)
plt.imshow(res)

# plt.show()
# plt.savefig('res.jpeg',res)

# pylab.show()

plt.show()
