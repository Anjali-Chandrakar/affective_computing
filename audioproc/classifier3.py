import numpy as np
import scipy.cluster.vq as vq
from scikits.audiolab import wavread
from sklearn import svm
from math import log,exp,isnan
import pickle

co=16

def sigmoid(x):
  return ((100.0) / (2 + exp(x)))

def check(t):
    for x in t:
        if isnan(x):
            return False
    return True

cn=0

def valu(c):
    if 'E' in c:
        #disgust
        return 0
    if 'T' in c:
        #sad
        return 1
    if 'F' in c:
        #joy
        return 2
    if 'A' in c:
        #fear
        return 3
    if 'W' in c:
        #anger
        return 4
    if 'N' in c:
        #neutral
        return 5
    return (-1)

def mean(c):
    ans=0.0
    cn=0
    for x in c:
        ans+=x
        cn+=1
    return (ans*(1.0)/cn)

def var(c,m):
    ans=0.0
    cn=0
    for x in c:
        ans+=((x-m)*(x-m))
        cn+=1
    return (ans*(1.0)/cn)

obs = pickle.load( open( "obs.pickle", "rb" ) )
vect = pickle.load( open( "features.pickle", "rb" ) )
tmt = pickle.load( open( "class.pickle", "rb" ) )

obs=np.array(obs)

print "good"

codes=[]

cn=0
tgt=[]
codu=[]

for j in range(0,len(vect)):
    cn+=1
    t=vect[j]
    i=0
    l=len(t)
    while ((i+100)<l):
        lst=[]
        kst=[]
        fg=0
        dicy={}
        
        for o in range(0,13):
            dicy[o]=0.0
            
        for x in t[i:i+100]:
            if (check(x)):
                for o in range(0,len(x)):
                    lst.append((round(x[o],2)))
                    dicy[o]+=x[o]
            else:
                fg=(-1)
        i+=100
        if (fg==(-1)):
            continue

        for o in range(0,13):
            dicy[o]=(dicy[o]/10)
            
        m=mean(lst)
        v=var(lst,m)
        mx=np.max(lst)
        mn=np.min(lst)

        lst.append((round(m,2)))
        lst.append((round(v,2)))
        lst.append(mx)
        lst.append(mn)

        kst.append((round(m,2)))
        kst.append((round(v,2)))
        kst.append(mx)
        kst.append(mn)

        for o in range(0,13):
            kst.append(round(dicy[o],2))
            lst.append(round(dicy[o],2))
                    
        tgt.append(tmt[j])
        codes.append(lst)
        codu.append(kst)

print "good"

mina=maxa=len(codes[0])

for x in codes:
    if (len(x)<mina):
        mina=len(x)
    if(len(x)>maxa):
        maxa=len(x)

sz=len(tgt)

#X = np.column_stack([vq.whiten(codes), tgt])
X=np.array(codes)

Y=[]

"""for a in X:
    ls=[]
    for t in a:
        ls.append(((t*(2.0))/co))
    Y.append(ls)"""

X=np.array(X)

tgt=np.array(tgt)

tr=X[0:((4*sz)/5)]
trg=tgt[0:((4*sz)/5)]
ts=X[((4*sz)/5):]
tsg=tgt[((4*sz)/5):]

n_classes=6

clf=svm.LinearSVC(class_weight='auto')

print tr[0:2]
print tgt[0:100]
print "total size: ",len(tgt)

clf.fit(tr,trg)

accu={}
cns={}
paccu={}
pcns={}

for i in range(0,6):
    accu[i]=0
    cns[i]=0
    paccu[i]=0
    pcns[i]=0


pred = clf.predict(tr)
print pred[0:100]
tsa = 0
cn=0
for i in range(0,len(tr)):
    cn+=1
    if (pred[i]==trg[i]):
        tsa+=1
        accu[pred[i]]+=1
        paccu[pred[i]]+=1
    pcns[pred[i]]+=1
    cns[trg[i]]+=1
    
print ((tsa*(1.0))/cn)

for i in range(0,6):
    print i,"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/pcns[i])
    print i,"recall : ",accu[i],cns[i],((accu[i]*(1.0))/cns[i])

pred = clf.predict(ts)
print pred[0:100]

for i in range(0,6):
    accu[i]=0
    cns[i]=0
    paccu[i]=0
    pcns[i]=0

tsa = 0
cn=0
for i in range(0,len(ts)):
    cn+=1
    if (pred[i]==tsg[i]):
        tsa+=1
        accu[pred[i]]+=1
        paccu[pred[i]]+=1
    pcns[pred[i]]+=1
    cns[tsg[i]]+=1

print ((tsa*(1.0))/cn)

for i in range(0,6):
    print i,"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/pcns[i])
    print i,"recall : ",accu[i],cns[i],((accu[i]*(1.0))/cns[i])


X=np.array(codu)

tgt=np.array(tgt)

tr=X[0:((4*sz)/5)]
trg=tgt[0:((4*sz)/5)]
ts=X[((4*sz)/5):]
tsg=tgt[((4*sz)/5):]

n_classes=6

clf=svm.LinearSVC(class_weight='auto')

print tr[0:2]
print tgt[0:100]
print "total size: ",len(tgt)

clf.fit(tr,trg)

pred = clf.predict(tr)
print pred[0:100]

for i in range(0,6):
    accu[i]=0
    cns[i]=0
    paccu[i]=0
    pcns[i]=0

tsa = 0
cn=0
for i in range(0,len(tr)):
    cn+=1
    if (pred[i]==trg[i]):
        tsa+=1
        accu[pred[i]]+=1
        paccu[pred[i]]+=1
    pcns[pred[i]]+=1
    cns[trg[i]]+=1

print ((tsa*(1.0))/cn)

for i in range(0,6):
    print i,"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/(pcns[i]+0.001))
    print i,"recall : ",accu[i],cns[i],((accu[i]*(1.0))/cns[i])

pred = clf.predict(ts)
print pred[0:200]

for i in range(0,6):
    accu[i]=0
    cns[i]=0
    paccu[i]=0
    pcns[i]=0
    
tsa = 0
cn=0
for i in range(0,len(ts)):
    cn+=1
    if (pred[i]==tsg[i]):
        tsa+=1
        accu[pred[i]]+=1
        paccu[pred[i]]+=1
    pcns[pred[i]]+=1
    cns[tsg[i]]+=1

print ((tsa*(1.0))/cn)

for i in range(0,6):
    print i,"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/(pcns[i]+0.001))
    print i,"recall : ",accu[i],cns[i],((accu[i]*(1.0))/cns[i])
