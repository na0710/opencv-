# import the necessary packages
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
from PIL import Image
"""
width_scale     = int(sys.argv[1])
scale_min       = float(sys.argv[2])
scale_max       = float(sys.argv[3])
scale_inc       = float(sys.argv[4])
# parameters      = sys.argv[5]
#output_file     = sys.argv[6]
#imageFileName   = sys.argv[7]
"""
width_scale     = 1.2
scale_min       = 1.0
scale_max       = 2.5
scale_inc       = 0.5   

def list_object(a,end,inc):
    rangei = []
    while (a<=end):
        rangei.append(a)
        a=a+inc;
    return rangei

def detect_object(rangei,rangej,img,hog):
    flag = 1
    rects_final = []
    for scale_image in rangei:
        image = imutils.resize(img, width=int(scale_image*img.shape[1]))
        for hog_scale in rangej:
            (rects, weighs) = hog.detectMultiScale(image, winStride=(3,3),padding=(6, 6), scale=hog_scale)
            if(len(rects)>0):
                if(flag):
                    rects_final = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])/scale_image
                    flag = 0
                else:
                    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])/scale_image
                    rects_final = np.vstack((rects_final,rects))
    return rects_final

param = []

img = cv2.imread("/home/nisha/opencv_start/digitimage.png",0)
img = img[:,0:960]
print(np.shape(img))

"""def resizing(ximage):
    image = Image.fromarray(ximage)
    def imresize(im,sz):
        return im.resize(sz)
    image1 = imresize(image,(650,650))
    image1_array = np.array(image1.getdata(),np.float64) 
    image1_array = np.reshape(image1_array,(650,650))
    return image1_array/255.0"""   

#img1 = resizing(img)
img_final = img.copy()
print(np.shape(img_final))

hog = cv2.HOGDescriptor()
param = np.transpose(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.setSVMDetector(param)

rangei = list_object(0.25,width_scale,0.05)
rangej = list_object(scale_min,scale_max,scale_inc)

pos_objects = []
rects = detect_object(rangei,rangej,img,hog)
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
i = 0

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img_final, (xA, yA), (xB, yB), (0, 255, 0), 2)
    pos_objects.append([xA, yA, xB, yB])
    img1 = img[yA:yB,xA:xB]
    cv2.imwrite("output/"+str(i)+".jpg",img1)
    i = i+1

np.save("position_output.npy",np.asarray(pos_objects,np.float32))
cv2.imshow("image",img_final)
cv2.waitKey(0)
