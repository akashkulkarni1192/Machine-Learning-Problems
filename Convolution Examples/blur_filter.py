import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the Lena image
img = mpimg.imread('lena.png')

# what does it look like?
plt.imshow(img)
plt.show()

# make it black and white
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# create a Gaussian blurring filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50.)

# filter looks like this
plt.imshow(W, cmap='gray')
plt.show()

# Convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

# after convolution, the output signal is N1 + N2 - 1

# we can also just make the output the same size as the input
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()
print(out.shape)


# in color
out3 = np.zeros(img.shape)
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
plt.imshow(out3)
plt.show() # does not look like anything
