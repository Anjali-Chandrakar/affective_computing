import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier

fc=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

vect=[]
tgt=[]

def valu(c):
    if 'HA' in c:
        return 1
    if 'SU' in c:
        return 2
    if 'AN' in c:
        return 3
    if 'NE' in c:
        return 4
    if 'DI' in c:
        return 5
    if 'FE' in c:
        return 6
    if 'SA' in c:
        return 7

cn=0

for fnm in os.listdir(os.getcwd())[0:150]:
    if cn%20==0:
        print cn
    cn+=1
    if '.py' in fnm:
        continue
    frame=cv2.imread(fnm)
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        dst=cv2.resize(rg,(25,25))
        lst=[]
        for x in dst:
            for y in x:
                lst.append(y[0])
        vect.append(lst)
        tgt.append(valu(fnm))
        cv2.imshow('f2',rc)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',dst)

print tgt

clf=(svm.LinearSVC(class_weight='auto'))
clf.fit(vect,tgt)

ans=0
cn=0

for fnm in os.listdir(os.getcwd())[150:]:
    tmp=[]
    print "cn: ",cn,
    if '.py' in fnm:
        continue
    frame=cv2.imread(fnm)
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        dst=cv2.resize(rg,(25,25))
        lst=[]
        for x in dst:
            for y in x:
                lst.append(y[0])
        tmp.append(lst)
    print "Val: ",valu(fnm),
    if (len(tmp)==0):
        continue
    if (len(tmp[0])!=625):
        continue
    cn+=1
    t=clf.predict(tmp)
    print "t: ",t
    if (valu(fnm) in t):
        ans+=1

print "\n"
print (ans*(1.0))/cn


cap=cv2.VideoCapture(0)

while(1):
    tmp=[]
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.8, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        dst=cv2.resize(rg,(25,25))
        lst=[]
        for x in dst:
            for y in x:
                lst.append(y)
        tmp.append(lst)
        cv2.imshow('f2',rc)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',dst)
    cv2.imshow('f1',frame)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
    elif (k==ord('s')):
        print "THIS IS ",clf.predict(tmp)

cap.release()

cv2.destroyAllWindows()
