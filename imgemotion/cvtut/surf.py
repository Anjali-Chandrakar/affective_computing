import cv2
import numpy as np

cap=cv2.VideoCapture(0)
surf=cv2.SURF(5000)

while(1):
    _, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kp, des = surf.detectAndCompute(gray,None)
    img2 = cv2.drawKeypoints(gray,kp,None,(255,0,0),4)
    cv2.imshow('f1',frame)
    cv2.imshow('f2',img2)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

