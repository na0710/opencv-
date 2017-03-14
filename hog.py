bin_n = 9 # Number of bins
import cv2
import numpy as np
import csv 
csvfile= open('features1.csv', 'w')
writer = csv.writer(csvfile)
hog_features = []
img=cv2.imread('pic71.jpg')
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
mag, ang = cv2.cartToPolar(gx, gy)
    
# quantizing binvalues in (0...16)
bins = np.int32(bin_n*ang/(2*np.pi))

# Divide to 4 sub-squares
bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
hist=np.hstack(hists)

writer.writerow(hist)
csvfile.close()

