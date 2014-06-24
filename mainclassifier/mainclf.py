import numpy as np

from scipy.io import loadmat
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

"""sgementaxis code.

This code has been implemented by Anne Archibald, and has been discussed on the
ML."""

import warnings

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap < 0 or length <= 0:
        raise ValueError, "overlap must be nonnegative and length must "\
                          "be positive"

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError, \
              "Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'"
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


import numpy as np

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def mel2hz(m):
    """Convert an array of frequency in Hz into mel."""
    return (np.exp(m / 1127.01048) - 1) * 700


def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def mfcc(input, nwin=256, nfft=512, fs=16000, nceps=13):
    """Compute Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed

    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.

    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum

    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    # MFCC parameters: taken from auditory toolbox
    over = nwin - 160
    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]

    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = np.abs(fft(framed, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

    return ceps, mspec, spec

def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)

"""if __name__ == '__main__':
    extract = loadmat('extract.mat')['extract']
    ceps = mfcc(extract)
"""

import nltk
import pickle
import cv2
import os
import random
from sklearn import svm
import pyaudio
import wave
import scipy.cluster.vq as vq
from scikits.audiolab import wavread
from sklearn import neighbors
from math import log,exp,isnan
from os import system
from time import sleep

def clean_audio(filename):
        cmd = []
        cmd[0] = "ffmpeg -i" + " " + filename + ".wav -vn -ss 00:00:00 -t 00:00:01 noiseaud.wav"
        cmd[1] = "sox noiseaud.wav -n noiseprof noise.prof"
        cmd[2] = "sox" + " " + filename + ".wav" + " " + filename+ "-clean.wav noisered noise.prof 0.26"

        for com in cmd:
                system(com)
                sleep(0.1)

#TEXT

print "INITIALISING PROGRAM..."

aneglist = pickle.load( open( "aneglist.pickle", "rb" ) )
aposlist = pickle.load( open( "aposlist.pickle", "rb" ) )
nneglist = pickle.load( open( "nneglist.pickle", "rb" ) )
nposlist = pickle.load( open( "nposlist.pickle", "rb" ) )
rneglist = pickle.load( open( "rneglist.pickle", "rb" ) )
rposlist = pickle.load( open( "rposlist.pickle", "rb" ) )
vneglist = pickle.load( open( "vneglist.pickle", "rb" ) )
vposlist = pickle.load( open( "vposlist.pickle", "rb" ) )
disco = pickle.load( open( "dicy.pickle", "rb" ) )

print "TEXT CLASSIFIER LOAD SUCCESFUL"

def valu(c,t):
    val=(0,0)
    tt=c
    if c in disco:
            if ('JJ' in t):
                val=(aposlist["%s" % tt],aneglist["%s" % tt])
            elif ('NN' in t):
                val=(nposlist["%s" % tt],nneglist["%s" % tt])
            elif ('RB' in t):
                return 0
            elif ('VB' in t):
                val=(vposlist["%s" % tt],vneglist["%s" % tt])
    return val

kernel = np.ones((2,2),np.float32)/4

fc=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
nc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml')
mc= cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')
lec=cv2.CascadeClassifier('C:\OpenCV\opencv\sources\data\haarcascades\haarcascade_lefteye_2splits.xml')

vect=[]
vect2=[]
tgt=[]

def mag(a):
        ans=0
	for i in a:
		ans=ans+(i*i)
	ans=sqrt(ans)
	if(ans==0):
                ans=1
	return ans

cn=0

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

def predictor(pos,neg,t):
    if (pos>(neg+0.3)):
        if (t==6):
            return 2
        elif (t==5 or t==7):
            return 4
        else:
            return t
    elif (neg>(pos+0.3)):
        if (t==2):
            return 6
        elif (t==4):
            return 7
        elif (t==3):
            return 5
        else:
            return t
    else:
        return t    
        

#OWN IMAGE

train=pickle.load(open("owntrain1.pickle", "rb" ) )
trgt=pickle.load(open("owntrgt1.pickle", "rb" ) )
train2=pickle.load(open("owntrain2.pickle", "rb" ) )
trgt2=pickle.load(open("owntrgt2.pickle", "rb" ) )

imgclf=(svm.LinearSVC(class_weight='auto'))
imgclf.fit(train,trgt)

imgclf2=(svm.LinearSVC(class_weight='auto'))
imgclf2.fit(train2,trgt2)


# GENERAL IMAGE

vecy=pickle.load(open( "vect.pickle", "rb" ) )

random.shuffle(vecy)

train=[]
trgt=[]

for i in vecy:
    train.append(i[1])
    trgt.append(i[0])

imgenclf=(svm.LinearSVC(class_weight='auto'))
imgenclf.fit(train,trgt)

print "FACIAL EXPRESSION CLASSIFIER LOAD SUCCESFUL"

#AUDIO

def sigmoid(x):
  return ((100.0) / (2 + exp(x)))

def check(t):
    for x in t:
        if isnan(x):
            return False
    return True

vect=[]
obs=[]

cn=0

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

tmt=[]

obs = pickle.load( open( "obs.pickle", "rb" ) )
vect = pickle.load( open( "features.pickle", "rb" ) )
tmt = pickle.load( open( "class.pickle", "rb" ) )

obs=np.array(obs)

codes=[]

cn=0
tgt=[]
codu=[]
tar=[]

for i in range(0,len(vect)):
    tar.append((vect[i],tmt[i]))

random.shuffle(tar)

for j in range(0,len(tar)):
    cn+=1
    t=tar[j][0]
    i=0
    l=len(t)
    while ((i+70)<l):
        lst=[]
        kst=[]
        fg=0
        dicy={}
        
        for o in range(0,13):
            dicy[o]=0.0
            
        for x in t[i:i+70]:
            if (check(x)):
                for o in range(0,len(x)):
                    lst.append((round(x[o],2)))
                    dicy[o]+=x[o]
            else:
                fg=(-1)
        i+=70
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
        lst.append((round(mx,2)))
        lst.append((round(mn,2)))

        kst.append((round(m,2)))
        kst.append((round(v,2)))
        kst.append((round(mx,2)))
        kst.append((round(mn,2)))

        for o in range(0,13):
            kst.append(round(dicy[o],2))
            lst.append(round(dicy[o],2))
                    
        tgt.append(tar[j][1])
        codes.append(lst)
        codu.append(kst)

sz=len(tgt)

X=np.array(codu)

tgt=np.array(tgt)

tr=X
trg=tgt

n_classes=7

audclf = neighbors.KNeighborsClassifier(7, weights='uniform')
audclf.fit(tr,trg)

print "AUDIO CLASSIFIER LOAD SUCCESFUL","\n"

#FUNCTION DEFINITIONS

def txtclas(flag=0):
    st=raw_input("Enter the sentence : ")
    wds=st.split()
    post=nltk.pos_tag(wds)
    psum=0
    nsum=0
    a=0
    b=0
    fg=0
    for i in range(0,len(wds)):
        t=post[i][1]
        tt=(post[i][0].lower())
        if tt in disco and fg==1:
            tup=valu(tt,t)
            if(tup!=0):
                if (a!=b):
                    psum+=(2*(a-b)*tup[0])
                    nsum+=(2*(a-b)*tup[1])
                    fg=0
                else:
                    psum+=(tup[0])
                    nsum+=(tup[1])
            else:
                a+=rposlist["%s" % tt]
                b+=rneglist["%s" % tt]
        elif tt in disco:
            tup=valu(tt,t)
            if(tup!=0):
                psum+=(tup[0])
                nsum+=(tup[1])
            else:
                a=rposlist["%s" % tt]
                b=rneglist["%s" % tt]
                fg=1

    if(flag==1):
        return (psum,nsum)
    
    tot=psum+nsum 
    print "On a scale of -1 to 1, these are the scores"
    print "Positive score : ",psum
    print "Negative score : ",nsum
    print "So you can say that the sentence is : ",
    if ((psum+0.5)<nsum):
        print "Very Negative"
    elif(psum<nsum):
        print "Negative"
    elif((nsum+0.5)<psum):
        print "Very Positive"
    elif(nsum<psum):
        print "Positive"
    else:
        print "Neutral"

def imgclas(pos,neg):

    xt=input("Are you ready? (Enter numbers) : ")
    print "Press 'q' to exit."
    if (xt==1):
        imgenclas(pos,neg)
    else:
        imgownclas(pos,neg)
    
def imgenclas(pos,neg):
    cap=cv2.VideoCapture(0)

    minc=0
    ninc=0
    eincp=0
    eincn=0
    inc=0

    while (1):
        _,frame=cap.read()
        frame = cv2.filter2D(frame,-1,kernel)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        tmp=gray
        nn,mm,ff,ee=0,0,0,0
        faces = fc.detectMultiScale(gray, 1.8, 5)
        mos = mc.detectMultiScale(frame, 3.0, 12)
        nos = nc.detectMultiScale(frame, 1.8, 5)
        lst=[]

        gray=cv2.equalizeHist(gray)
   
        cv2.imshow('f5',gray)
                
        if(len(nos)!=1):
            print "Position yourself correctly"
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
            leyes = lec.detectMultiScale(rg,1.10,4)
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
            print "Position yourself correctly"
            continue
        k=cv2.waitKey(10)
        if k==ord('q'):
            break
        print "You are feeling :",antival(predictor(pos,neg,imgenclf.predict(lst)))

        cap.release()

        cv2.destroyAllWindows()

def imgownclas(pos,neg):
    cap=cv2.VideoCapture(0)

    minc=0
    ninc=0
    eincp=0
    eincn=0
    inc=0
    while (1):
        _,frame=cap.read()
        frame = cv2.filter2D(frame,-1,kernel)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        tmp=gray
        nn,mm,ff,ee=0,0,0,0
        faces = fc.detectMultiScale(gray, 1.8, 5)
        mos = mc.detectMultiScale(frame, 3.0, 12)
        nos = nc.detectMultiScale(frame, 1.8, 5)
        lst=[]
        tax=[]

        gray=cv2.equalizeHist(gray)
        cv2.imshow('f5',frame)
        #cv2.imshow('f5',gray)
            
        if(len(nos)!=1):
            print "Position yourself correctly"
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
                    tax.append(wy)
     
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
                    tax.append(wy)
    
        for (x,y,w,h) in faces:
            rg=tmp[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            rg=cv2.equalizeHist(rg)
            leyes = lec.detectMultiScale(rg,1.08,4)
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
                """cv2.namedWindow('f3', cv2.WINDOW_NORMAL)
                cv2.imshow('f3',drg)"""
        cv2.imshow('f4',frame)
    
        if(nn!=1 or mm!=1 or ff!=1):
            print "Position yourself correctly"
            continue

        k=cv2.waitKey(500)
        if k==ord('q'):
            break    
    
        if((ee!=2)):
            print "You are feeling : ",antival(predictor(pos,neg,imgclf2.predict(tax)))
            continue
            
        print "You are feeling : ",antival(predictor(pos,neg,imgclf.predict(lst)))

    cap.release()

    cv2.destroyAllWindows()

chunk=1024
    
def getFileReady(name):
    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16,channels = 1,rate = 44100,input = True, frames_per_buffer = chunk)
    wf = wave.open(name+'.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    return (p,stream,wf)
    
def audclas():
    c='y'
    
    while (c=='y' or c=='Y'):
       	(p,stream,wf)=getFileReady("temp")
        i=0
        print "Recording in progress"
	while i<200:
		data = stream.read(chunk)
		wf.writeframes(data)
		i+=1
	stream.close()
	p.terminate()
	wf.close()
	#clean_audio("temp")
	#data, fs = wavread("temp-clean.wav")[:2]
        data, fs = wavread("temp.wav")[:2]
        ceps = mfcc(data, fs=fs)[0]
        ceps=np.array(ceps)
        ceps=vq.whiten(ceps)
        t=ceps
        l=len(ceps)
        i=0
        while ((i+70)<l):
          lst=[]
          kst=[]
          fg=0
          dicy={}
          
          for o in range(0,13):
              dicy[o]=0.0
            
          for x in t[i:i+70]:
              if (check(x)):
                  for o in range(0,len(x)):
                      lst.append((round(x[o],2)))
                      dicy[o]+=x[o]
              else:
                  fg=(-1)
          i+=70
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
          lst.append((round(mx,2)))
          lst.append((round(mn,2)))
  
          kst.append((round(m,2)))
          kst.append((round(v,2)))
          kst.append((round(mx,2)))
          kst.append((round(mn,2)))   

          for o in range(0,13):
            kst.append(round(dicy[o],2))
            lst.append(round(dicy[o],2))

          print "You are feeling :",antival(audclf.predict(kst))
        c=raw_input("Do you want to continue? (Enter y or n) : ")

def imgtxtclas():
    (psum,nsum)=txtclas(1)
    print "On the basis of only text, you are feeling : ",
    if ((psum+0.5)<nsum):
        print "Very Negative"
    elif(psum<nsum):
        print "Negative"
    elif((nsum+0.5)<psum):
        print "Very Positive"
    elif(nsum<psum):
        print "Positive"
    else:
        print "Neutral"
    imgclas(psum,nsum)
    
#PRECISION VALUES

imgenpres=[0.93,0.56,0.59,0.81,0.67,0.45,0.43]
imgownpres=[1,1,0.8,0.8,0.8,0.9,0.9]
audpres=[0.25,0.15,0.14,0,0,0.33,0.25]
          

#MENU

ch=0

while (ch!=7):
    print "\nEmotion Recogniser"
    print "Select one of the following commands by entering the appropriate number :"
    print "1. Classification on the basis of text."
    print "2. Classification on the basis of facial expression. Takes input from the webcam."
    print "3. Classification on the basis of speech."
    print "4. Classification on the basis of visual and audio features."
    print "5. Classification on the basis of visual and text features."
    print "6. Classification on the basis of all the features."
    print "7. Exit"

    ch=input("Give your choice : ")

    if (ch==1):
        print "This is the text classifier."
        c='y'
        while ((c=='y') or (c=='Y')):
            txtclas()
            c=raw_input("Do you want to continue? (Enter y or n)")

    elif (ch==2):
        print "This is the facial expression classifier."
        imgclas(0,0)

    elif (ch==3):
        print "This is the audio classifier."        
        audclas()

    elif (ch==4):
        print "This is the visual-audio classifier."
        imgclas(0,0)

    elif (ch==5):
        print "This is the visual-text classifier."
        imgtxtclas()

    elif (ch==6):
        print "This classifier takes all the features."
        imgtxtclas()
        
        
        
        
