import cv2
import numpy as np
import os
from numpy import histogram
from pylab import interp
from sklearn.decomposition import PCA
from math import sqrt

fc=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

thresh=150

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
    
def histeq(im,nbr_bins=127):

   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape)

cn=0

for fnm in os.listdir(os.getcwd())[0:150]:
    if cn%20==0:
        print cn
    cn+=1
    if '.py' in fnm:
        continue
    frame=cv2.imread(fnm,0)
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    for (x,y,w,h) in faces:
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        dst=cv2.resize(rg,(280,180))
        lst=[]
        nst= cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)[1]
        #nst=histeq(im)
        for x in nst:
            for y in x:
                lst.append(y)
        vect.append(lst)
        tgt.append(valu(fnm))      
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',nst)
    cv2.imshow('f1',frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

print tgt

cv2.destroyAllWindows()

x=np.array(vect)

pca=PCA()
pca.fit(x)
tran=pca.transform(x)
mags={}
for i in range(0,len(tran)):
    mags[i]=mag(tran[i])

print (tran[0])

cn=0
accu=0

def eucl(a,b,c,d):
    ans=0
    for i in range(0,len(a)):
        #ans+=(((a[i]/b)-(c[i]/d))*((a[i]/b)-(c[i]/d)))
        ans+=((a[i]-c[i])*(a[i]-c[i]))
    return ans

def classify(lst):
    dist={}
    cats={}
    lmag=mag(lst)
    for i in range(0,len(tran)):
        dist[i]=eucl(lst,lmag,tran[i],mags[i])
    lt=sorted(dist,key=dist.__getitem__)[0:7]
    for i in range(1,8):
        cats[i]=0
    for key in lt:
        cats[tgt[key]]+=1
    mx=0
    ans=0
    for i in range(1,8):
        if(cats[i]>mx):
            ans=i
            mx=cats[i]
    return ans

def classi(lst):
    dist={}
    cats={}
    lmag=mag(lst)
    for i in range(0,len(tran)):
        dist[i]=eucl(lst,lmag,tran[i],mags[i])
    lt=sorted(dist,key=dist.__getitem__)[0:7]
    for i in range(1,8):
        cats[i]=0
    for key in lt:
        cats[tgt[key]]+=1
    mx=0
    ans=0
    for i in range(1,8):
        print i,((cats[i]*(1.0))/7)
        if(cats[i]>mx):
            ans=i
            mx=cats[i]
    return ans

acs={}
cns={}

for i in range(1,8):
    acs[i]=0
    cns[i]=0

for fnm in os.listdir(os.getcwd())[150:]:
    if cn%20==0:
        print cn
    if '.py' in fnm:
        continue
    frame=cv2.imread(fnm,0)
    gray=frame
    faces = fc.detectMultiScale(gray, 1.8, 5)
    cst=[]
    for (x,y,w,h) in faces:
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        dst=cv2.resize(rg,(280,180))
        mn=0
        lst=[]
        nst = cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)[1]
        #nst=histeq(im)        
        for x in nst:
            for y in x:
                lst.append(y-mn)
        cst.append(lst)
    if len(faces)==0:
        continue
    rlst=pca.transform(cst)
    ans=classify(rlst[0])
    print "correct :",valu(fnm),"given ;",ans
    if(ans==valu(fnm)):
        accu+=1
        acs[valu(fnm)]+=1
    cns[valu(fnm)]+=1
    cn+=1

print "ov:",(accu*(1.0)/cn)

for i in range(1,8):
    print "ac:",i,acs[i],cns[i],(acs[i]*(1.0)/cns[i])

input()
    
cap=cv2.VideoCapture(0)

while(1):
    tmp=[]
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.5, 4)
    cst=[]
    for (x,y,w,h) in faces:
        rg = gray[y:y+h,x:x+w]
        rc = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        dst=cv2.resize(rg,(280,180))
        lst=[]
        nst = cv2.threshold(dst, thresh, 255, cv2.THRESH_OTSU)[1]
        cv2.imshow('f4',nst)
        #nst=histeq(dst)
        for x in nst:
            for y in x:
                lst.append(y)
        cst.append(lst)
        cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
        cv2.imshow('f3',nst)
    cv2.imshow('f1',frame)
    cv2.imshow('f2',gray)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
    elif (k==ord('s')):
        if len(faces)==0:
            continue
        rlst=pca.transform(cst)
        ans=classi(rlst[0])
        print "THIS IS ",ans

cap.release()

cv2.destroyAllWindows()
