import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pic11.jpg')

 
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)

# sure background area
sure_bg = cv2.dilate(thresh,kernel,iterations=3)


# Finding sure foreground area
dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)


# Finding unknown region
#sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
dst[markers == -1] = [255,0,0]

cv2.imshow('image',dst);
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('pic11.jpg',dst)
    cv2.destroyAllWindows()


