import numpy as np
import scipy.cluster.vq as vq
from scikits.audiolab import wavread
from sklearn import svm
from math import log,exp
import pickle

co=16

def sigmoid(x):
  return ((100.0) / (2 + exp(x)))

def check(t):
    for x in t:
        if(x>(2*co)) or (x<0):
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

obs = pickle.load( open( "obs.pickle", "rb" ) )
vect = pickle.load( open( "features.pickle", "rb" ) )
tmt = pickle.load( open( "class.pickle", "rb" ) )

obs=np.array(obs)

print "good"

cdb=vq.kmeans(obs,(2*co))[0]

codes=[]

print "good"

cn=0
tgt=[]

for j in range(0,len(vect)):
    cn+=1
    x=vect[j]
    x=vq.whiten(x)
    t=vq.vq(x,cdb)[0]
    i=0
    l=len(t)
    if (check(t)):
        while ((i+50)<l):
            codes.append(t[i:i+50])
            i+=50
            tgt.append(tmt[j])

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

for a in X:
    ls=[]
    for t in a:
        ls.append(((t*(2.0))/co))
    Y.append(ls)

X=np.array(X)

tgt=np.array(tgt)

tr=X[0:((4*sz)/5)]
trg=tgt[0:((4*sz)/5)]
ts=X[((4*sz)/5):]
tsg=tgt[((4*sz)/5):]

n_classes=6

print tr[0:3]

print tgt[0:50]

clf=svm.LinearSVC(class_weight='auto')

clf.fit(tr,trg)

pred = clf.predict(tr)
print pred[0:200]
train_accuracy = np.mean(pred.ravel() == trg.ravel()) * 100
print train_accuracy

pred = clf.predict(ts)
print pred[0:200]
train_accuracy = np.mean(pred.ravel() == tsg.ravel()) * 100
print train_accuracy

