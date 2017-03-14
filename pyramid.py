import cv2
import numpy as np,sys

A = cv2.imread('pic39.jpg')
B = cv2.imread('pic32.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
xrange=[0,1,2,3,4,5]
for i in xrange:
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange:
    G = cv2.pyrDown(G)
    gpB.append(G)

#xr=[6,5,4,3,2,1]

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in [5,4,3,2,1]:
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in [5,4,3,2,1]:
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange:
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)