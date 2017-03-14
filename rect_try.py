from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
print(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imageFileName = input("enter the name of the image file: ")
img = cv2.imread(imageFileName,1)
image = imutils.resize(img, width=2*img.shape[1])
(rects, weighs) = hog.detectMultiScale(image, winStride=(3,3),padding=(6, 6), scale=1.05)
orig = image.copy()
v=0;
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
	cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)
	roi=orig[yA:yB,xA:xB]
	cv2.imwrite("output/"+str(v)+".png",roi)
	v=v+1;
cv2.imwrite("output/"+str(5)+".png",orig)

		

    

