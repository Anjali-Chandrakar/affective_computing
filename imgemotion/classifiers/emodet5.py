import cv2
import numpy as np
import os
import random
from sklearn import svm

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
    
def mag(a):
        ans=0
	for i in a:
		ans=ans+(i*i)
	ans=sqrt(ans)
	if(ans==0):
                ans=1
	return ans

def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape)

cn=0

minc=0
ninc=0
einc=0

for fnm in os.listdir(os.getcwd()):
    if cn%20==0:
        print cn
    cn+=1
    if '.py' in fnm:
        continue
    nn,mm,ff,ee=0,0,0,0
    frame=cv2.imread(fnm,0)
    frame = cv2.filter2D(frame,-1,kernel)
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    mos = mc.detectMultiScale(frame, 2.8, 10)
    nos = nc.detectMultiScale(frame, 2.8, 5)
    lst=[]
    gray = cv2.filter2D(gray,-1,kernel)
    gray=cv2.equalizeHist(gray)
    if(len(nos)!=1):
        continue

    if(len(faces)==0):
        continue
    
    for (tx,ty,tw,th) in nos:
        #cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,125),2)
        drg = frame[ty:ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(60,60))
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
        #cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,127,0),2)
        drg = frame[min(b+d,ty):ty+th,tx:tx+tw]
        drg=cv2.equalizeHist(drg)
        dst=cv2.resize(drg,(40,80))
        nst=dst
        #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
        """cv2.namedWindow('f2', cv2.WINDOW_NORMAL)
        cv2.imshow('f2',nst)"""
        for wx in nst:
            for wy in wx:
                lst.append(wy)

    for (x,y,w,h) in faces:
        rg=gray[y:y+h,x:x+w]
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        leyes = lec.detectMultiScale(rg,1.04,4)
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        nn,mm,ff,ee=len(nos),len(nmos),len(faces),len(leyes)
        #print len(nos),len(nmos),len(faces)," : ",len(leyes)
        if(len(leyes)!=2):
            einc+=1
        if(len(nmos)!=1):
            minc+=1
        if(len(nos)!=1):
            ninc+=1
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(0,255,0),4)
            drg = rg[ey:ey+eh,ex:ex+ew]
            dst=cv2.resize(drg,(40,40))
            nst=dst
            #nst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)[1]
            #nst=histeq(im)
            lst=[]
            for x in nst:
                for y in x:
                    lst.append(y)
        """cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',rg)"""
    if(nn!=1 or mm!=1 or ee!=2 or ff!=1):
        continue
    vect.append([valu(fnm),lst])
    #cv2.imshow('f1',frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

print "RESULT:",minc,ninc,einc

cv2.destroyAllWindows()

random.shuffle(vect)

train=[]
trgt=[]
test=[]
tegt=[]

for i in vect[0:130]:
    train.append(i[1])
    trgt.append(i[0])

for i in vect[130:]:
    test.append(i[1])
    tegt.append(i[0])

clf=(svm.LinearSVC(class_weight='auto'))
clf.fit(train,trgt)

acs={}
cns={}
accu=0
cn=0

for i in range(1,8):
    acs[i]=0
    cns[i]=0

for i in range(0,len(test)):
    ans=clf.predict(test[i])
    print "correct :",tegt[i],"given ;",ans
    if(ans==tegt[i]):
        accu+=1
        acs[tegt[i]]+=1
    cns[tegt[i]]+=1
    cn+=1

print "ov:",(accu*(1.0)/cn)

for i in range(1,8):
    print "ac:",i,acs[i],cns[i],(acs[i]*(1.0)/cns[i]),"\t\t",(((accu-acs[i])*(1.0))/(cn-cns[i]))

input()

cap=cv2.VideoCapture(0)

while(1):
    _,frame=cap.read()
    frame = cv2.filter2D(frame,-1,kernel)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tmp=gray
    nn,mm,ff,ee=0,0,0,0
    faces = fc.detectMultiScale(gray, 1.8, 5)
    mos = mc.detectMultiScale(frame, 2.8, 10)
    nos = nc.detectMultiScale(frame, 2.8, 5)
    lst=[]

    gray=cv2.equalizeHist(gray)
   
    cv2.imshow('f5',gray)
            
    if(len(nos)!=1):
        continue
    
    for (tx,ty,tw,th) in nos:
        cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,125),2)
        drg = gray[ty:ty+th,tx:tx+tw]
        dst=cv2.resize(drg,(60,60))
        nst=dst
        nst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)[1]
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
        cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,127,0),2)
        drg = gray[ty:ty+th,tx:tx+tw]
        dst=cv2.resize(drg,(20,40))
        nst=dst
        nst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.namedWindow('f2', cv2.WINDOW_NORMAL)
        cv2.imshow('f2',nst)
        for wx in nst:
            for wy in wx:
                lst.append(wy)

    for (x,y,w,h) in faces:
        rg=tmp[y:y+h,x:x+w]
        rg=cv2.equalizeHist(rg)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        leyes = lec.detectMultiScale(rg,1.04,4)
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(rg,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        nn,mm,ff,ee=len(nos),len(nmos),len(faces),len(leyes)
        #print len(nos),len(nmos),len(faces)," : ",len(leyes)
        if(len(leyes)!=2):
            einc+=1
        if(len(nmos)!=1):
            minc+=1
        if(len(nos)!=1):
            ninc+=1
        for (ex,ey,ew,eh) in leyes:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            drg = rg[ey:ey+eh,ex:ex+ew]
            dst=cv2.resize(drg,(40,40))
            nst=dst
            nst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)[1]
            cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
            cv2.imshow('f3',nst)
            #nst=histeq(im)
            lst=[]
            for x in nst:
                for y in x:
                    lst.append(y)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',frame)
        cv2.namedWindow('f4', cv2.WINDOW_NORMAL)
        cv2.imshow('f4',rg)
    if(nn!=1 or mm!=1 or ee!=2 or ff!=1):
        continue
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
    elif (k==ord('s')):
        print "THIS IS ",clf.predict(lst)

cap.release()

cv2.destroyAllWindows()

