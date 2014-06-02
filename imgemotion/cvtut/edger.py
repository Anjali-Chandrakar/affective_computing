import cv2
import numpy as np
from matplotlib import pyplot as plt

cap=cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

while(1):
    _, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame,100,200)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    laplacian=cv2.Laplacian(gray,cv2.CV_64F)
    #cv2.imshow('sox',sobelx)
    cv2.imshow('lap',laplacian)
    cv2.imshow('frm',edges)
    cv2.imshow('frame',frame)
    cv2.imshow('fr',gray)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

