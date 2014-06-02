import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
    _, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    frame[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('f1',frame)
    cv2.imshow('f2',dst)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

