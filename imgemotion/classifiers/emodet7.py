import cv2
import numpy as np
import os
import random
from sklearn import svm
from sklearn.decomposition import PCA

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
kernel = np.ones((2,2),np.float32)/4

fc=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
nc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml')
mc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')
lec=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_lefteye_2splits.xml')

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
    return (-1)

def valy(c):
    if 'h' in c:
        return 1
    if 's' in c:
        return 2
    if 'a' in c:
        return 3
    if 'n' in c:
        return 4
    if 'd' in c:
        return 5
    if 'f' in c:
        return 6
    if 'sa' in c:
        return 7
    return (-1)

def antival(c):
    if (c==1):
        return 'Happy'
    elif (c==2):
        return 'Surprised'
    elif (c==3):
        return 'Angry'
    elif (c==4):
        return 'Neutral'
    elif (c==5):
        return 'Disgusted'
    elif (c==6):
        return 'Afraid'
    elif (c==7):
        return 'Sad'

def mag(a):
        ans=0
	for i in a:
		ans=ans+(i*i)
	ans=sqrt(ans)
	if(ans==0):
                ans=1
	return ans

cn=0

minc=0
ninc=0
eincp=0
eincn=0
inc=0
"""
for fnm in os.listdir(os.getcwd()):
    if '.py' in fnm or 'pickle' in fnm:
        continue
    if (valy(fnm[1:])==(-1) and valu(fnm)==(-1)):
        continue
    nn,mm,ff,ee=0,0,0,0
    frame=cv2.imread(fnm,0)
    frame = cv2.filter2D(frame,-1,kernel)
    tmp=frame
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    mos = mc.detectMultiScale(frame, 3.0, 13)
    nos = nc.detectMultiScale(frame, 2.8, 5)
    lst=[]
    
    if(len(nos)!=1):
        continue

    if(len(faces)==0):
        continue
    
    for (tx,ty,tw,th) in nos:
        #cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,125),2)
        drg = tmp[ty:ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(15,15))
        nst=dst
        #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
        cv2.namedWindow('f1', cv2.WINDOW_NORMAL)
        cv2.imshow('f1',nst)
        for wx in nst:
            for wy in wx:
                lst.append(wy)
     
    nmos=[]
    (a,b,c,d)=nos[0]
    
    for (tx,ty,tw,th) in mos:
        if(ty>=b):
            nmos.append([tx,ty,tw,th])

    for (tx,ty,tw,th) in nmos:
        #cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,127,0),2)
        drg = tmp[min(b+d,ty):ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(10,20))
        nst=dst
        #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
        cv2.namedWindow('f2', cv2.WINDOW_NORMAL)
        cv2.imshow('f2',nst)
        for wx in nst:
            for wy in wx:
                lst.append(wy)
    
    for (x,y,w,h) in faces:
        rg=tmp[y:y+h,x:x+w]
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg=cv2.equalizeHist(rg)
        leyes = lec.detectMultiScale(rg,1.07,3)
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        nn,mm,ff,ee=len(nos),len(nmos),len(faces),len(leyes)
        #print len(nos),len(nmos),len(faces)," : ",len(leyes)
        if(len(leyes)>2):
            eincp+=1
            inc+=1
        elif (len(leyes)<2):
            eincn+=1
            inc+=1
        if(len(nmos)!=1):
            minc+=1
            inc+=1
        if(len(nos)!=1):
            ninc+=1
            inc+=1
        for (ex,ey,ew,eh) in leyes:
            #cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(0,255,0),4)
            drg = rg[ey:ey+eh,ex:ex+ew]
            drg=cv2.equalizeHist(drg)
            dst=cv2.resize(drg,(15,15))
            nst=dst
            #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
            #nst=histeq(im)
            for x in nst:
                for y in x:
                    lst.append(y)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',rg)
    cv2.imshow('f4',frame)
    cv2.imshow('f5',tmp)
    if(nn!=1 or mm!=1 or ee!=2 or ff!=1):
        continue
    if (valu(fnm)!=(-1)):
        vect.append([valu(fnm),lst])
    elif (valy(fnm[1:])!=(-1)):
        vect.append([valy(fnm[1:]),lst])

    if cn%20==0:
        print cn
    cn+=1

    #cv2.imshow('f1',frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

print "RESULT:",inc,minc,ninc,eincn,eincp

print "len",len(vect[0][1]),len(vect)

cv2.destroyAllWindows()"""


import pickle

vect=pickle.load(open( "features.pickle", "rb" ) )

print "good"

random.shuffle(vect)

train=[]
trgt=[]
test=[]
tegt=[]

for i in vect[0:170]:
    train.append(i[1])
    trgt.append(i[0])

for i in vect[170:]:
    test.append(i[1])
    tegt.append(i[0])

clf=(svm.LinearSVC(class_weight='auto'))
clf.fit(train,trgt)

acs={}
cns={}
pacs={}
pcns={}
aacs={}
acns=0
accu=0
cn=0

mat=np.zeros((8,8))
        
for i in range(1,8):
    acs[i]=0
    cns[i]=0
    pacs[i]=0
    pcns[i]=0
    aacs[i]=0
    
print tegt

for i in range(0,len(test)):
    ans=clf.predict(test[i])
    print "correct :",antival(tegt[i]),"given ;",antival(ans)
    acns+=1
    mat[tegt[i]][ans[0]]+=1
    if(ans[0]==tegt[i]):
        accu+=1
        acs[tegt[i]]+=1
        pacs[tegt[i]]+=1
    pcns[ans[0]]+=1
    cns[tegt[i]]+=1
    cn+=1

print "ov:",(accu*(1.0)/cn)

for i in range(1,8):
    print "recall :",antival(i),acs[i],cns[i],(acs[i]*(1.0)/cns[i])
    print "precision :",antival(i),pacs[i],pcns[i],(pacs[i]*(1.0)/pcns[i]),"\n"

print "Confusion matrix\n"

print antival(1), " : "
 
for i in range(1,8):
    print antival(i),

print "\n"

print mat


input()

cap=cv2.VideoCapture(0)

while(1):
    _,frame=cap.read()
    frame = cv2.filter2D(frame,-1,kernel)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tmp=gray
    nn,mm,ff,ee=0,0,0,0
    faces = fc.detectMultiScale(gray, 1.8, 5)
    mos = mc.detectMultiScale(frame, 1.8, 10)
    nos = nc.detectMultiScale(frame, 1.8, 5)
    lst=[]

    gray=cv2.equalizeHist(gray)
   
    cv2.imshow('f5',gray)
            
    if(len(nos)!=1):
        continue
    
    for (tx,ty,tw,th) in nos:
        cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,125),2)
        drg = tmp[ty:ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(15,15))
        nst=dst
        #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
        """cv2.namedWindow('f1', cv2.WINDOW_NORMAL)
        cv2.imshow('f1',nst)"""
        for wx in nst:
            for wy in wx:
                lst.append(wy)
     
    nmos=[]
    (a,b,c,d)=nos[0]
    
    for (tx,ty,tw,th) in mos:
        if(ty>=b):
            nmos.append([tx,ty,tw,th])

    for (tx,ty,tw,th) in nmos:
        cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,127,0),2)
        drg = tmp[min(b+d,ty):ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(10,20))
        nst=dst
        #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
        """cv2.namedWindow('f2', cv2.WINDOW_NORMAL)
        cv2.imshow('f2',nst)"""
        for wx in nst:
            for wy in wx:
                lst.append(wy)
    
    for (x,y,w,h) in faces:
        rg=tmp[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        rg=cv2.equalizeHist(rg)
        leyes = lec.detectMultiScale(rg,1.06,4)
        """for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)"""
        nn,mm,ff,ee=len(nos),len(nmos),len(faces),len(leyes)
        #print len(nos),len(nmos),len(faces)," : ",len(leyes)
        if(len(leyes)>2):
            eincp+=1
            inc+=1
        elif (len(leyes)<2):
            eincn+=1
            inc+=1
        if(len(nmos)!=1):
            minc+=1
            inc+=1
        if(len(nos)!=1):
            ninc+=1
            inc+=1
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(0,255,0),4)
            drg = rg[ey:ey+eh,ex:ex+ew]
            drg=cv2.equalizeHist(drg)
            dst=cv2.resize(drg,(15,15))
            nst=dst
            #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
            #nst=histeq(im)
            for x in nst:
                for y in x:
                    lst.append(y)
            cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
            cv2.imshow('f3',drg)
    cv2.imshow('f4',frame)
    if(nn!=1 or mm!=1 or ee!=2 or ff!=1):
        continue
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
    print "THIS IS ",antival(clf.predict(lst))

cap.release()

cv2.destroyAllWindows()

