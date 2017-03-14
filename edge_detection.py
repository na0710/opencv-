import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pic39.jpg')
edges = cv2.Canny(img,100,300)
plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), 
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

