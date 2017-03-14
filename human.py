from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import os
import csv
import sys
import imutils
import cv2

width_scale     = int(sys.argv[1])
scale_min       = float(sys.argv[2])
scale_max       = float(sys.argv[3])
scale_inc       = float(sys.argv[4])
parameters      = sys.argv[5]
output_file     = sys.argv[6]
imageFileName   = sys.argv[7]

def list_object(a,end,inc):
    rangei = []
    while (a<=end):
        rangei.append(a)
        a=a+inc;
    return rangei

def detect_object(rangei,rangej,img,hog):
    flag = 1
    for scale_image in rangei:
        image = imutils.resize(img, width=scale_image*img.shape[1])
        for hog_scale in rangej:
            (rects, weighs) = hog.detectMultiScale(image, winStride=(3,3),padding=(6, 6), scale=hog_scale)
            if(flag):
                rects_final = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])/scale_image
                flag = 0
            else:
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])/scale_image
                rects_final = np.vstack((rects_final,rects))
    return rects_final

parameters = np.load(parameters+".npy")

img = cv2.imread(imageFileName,1)
img_final = img.copy()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(parameters)


rangei = list_object(1,width_scale,1)
rangej = list_object(scale_min,scale_max,scale_inc)

pos_objects = []
rects = detect_object(rangei,rangej,img,hog)
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img_final, (xA, yA), (xB, yB), (0, 255, 0), 2)
    pos_objects.append([xA, yA, xB, yB])

np.save("output/"+output_file,np.asarray(pos_objects,np.float32))
cv2.imwrite("output/"+output_file+".png",img_final)
    

