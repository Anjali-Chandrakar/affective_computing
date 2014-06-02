import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
    _, frame=cap.read()
    #frame = cv2.blur(fr,(5,5))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = cv2.medianBlur(gray,5)
    cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,25,param1=60,param2=28,minRadius=10,maxRadius=30)

    if circles is None:
        continue

    circles = np.uint16(np.around(circles))
    
    for i in circles:
        print len(i)
        for j in i:
            cv2.circle(cimg,(j[0],j[1]),j[2],(0,255,0),2)
            cv2.circle(cimg,(j[0],j[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.imshow('frame',gray)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

