#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

SZ=20
bin_n = 9 # Number of bins
image=cv2.imread(input("Enter the image :"),1)


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     
    return hist

print(hog(image))

"""

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
print(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img=cv2.imread('pic71.jpg')
rangei = [5,4,3,2,1]
rangex = [1.1,1.2,1.3,1.4,1.5]
#image = imutils.resize(image, width=(800, image.shape[1]))
for i in rangei:
	image = imutils.resize(img, width=i*img.shape[1])
	print(i)
	# detect people in the image
	for x in rangex:
		(rects, weights) = hog.detectMultiScale(image, winStride=(3,3),padding=(6, 6), scale=x)
		orig = image.copy()
		# draw the original bounding boxes
		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 
		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)
		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))
		 
		# show the output images
		#cv2.imshow("Before NMS", orig)
		cv2.imwrite("output/"+str(i)+"_"+str(x)+".png",orig)
"""