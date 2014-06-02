import cv2
import numpy as np

cap=cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

while(1):
    _, frame=cap.read()
    #frame = cv2.blur(fr,(5,5))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lb=np.array([110,90,90])
    ub=np.array([119,255,255])
    lr=np.array([0,150,100])
    ur=np.array([5,250,200])
    msk1=cv2.inRange(hsv,lb,ub)
    msk2=cv2.inRange(hsv,lr,ur)
    res1=cv2.add(msk1,msk2)
    #ope = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
    res=cv2.bitwise_and(frame,frame,mask=res1)
    cv2.imshow('frame',frame)
    cv2.imshow('f5',res1)
    cv2.imshow('f3',res)
    #cv2.imshow('f1',ope)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

