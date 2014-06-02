import cv2
import numpy as np

cap=cv2.VideoCapture(0)
fc=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
ec = cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_eye.xml')
nc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml')
mc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')

while(1):
    _,frame=cap.read()
    #frame=cv2.imread("KA.AN1.39.tiff")
    #gray=frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.8, 5)
    mos = mc.detectMultiScale(gray, 1.8, 15)
    nos = nc.detectMultiScale(gray, 1.8, 10)
    for (x,y,w,h) in nos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,125),2)
    for (x,y,w,h) in mos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,127,0),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        eyes = ec.detectMultiScale(rg,2.5,1)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(rc,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        dst=cv2.resize(rg,(25,25))
        cv2.imshow('f2',rc)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',dst)
    cv2.imshow('f1',frame)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

