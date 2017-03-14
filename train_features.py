import sys
import csv
import pandas as pd
import numpy as np
import numpy as np
import cv2 
import os
bin_n = 9

features = []
responses = []


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     
    return hist


columns = ['name'];
pos_image = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Train/pos.csv'), sep='\t', names=columns,encoding='latin-1');

length_pos = len(pos_image)

i=0
while(i<length_pos):
    img = cv2.imread(pos_image.name[i])
    hog_features = hog(img)
    features.append(hog_features)
    responses.append(1)
    i=i+1

neg_image = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Train/neg.csv'), sep='\t', names=columns,encoding='latin-1');

length_neg = len(neg_image)

while(i<(length_pos+length_neg)):
    img = cv2.imread(neg_image.name[i-length_pos])
    hog_features = hog(img)
    features.append(hog_features)
    responses.append(0)
    i=i+1

np.save('features.npy',np.asarray(features,np.float32))

np.save('responses.npy',np.asarray(responses,np.float32))




