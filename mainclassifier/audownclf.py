import numpy as np

from scipy.io import loadmat
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

"""sgementaxis code.

This code has been implemented by Anne Archibald, and has been discussed on the
ML."""

import numpy as np
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





co=32

import os
import scipy.cluster.vq as vq
from scikits.audiolab import wavread
from sklearn import neighbors
from math import log,exp,isnan
import pickle
import random

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

tmt=[]
"""
for fnm in os.listdir(os.getcwd()):
    if '.py' in fnm:
        continue
    if(cn%10==0):
        print cn
    if(valu(fnm)==(-1)):
        continue
    cn+=1
    nn,mm,ff,ee=0,0,0,0
    data, fs = wavread(fnm)[:2]
    
    # ceps: cepstral cofficients
    ceps = mfcc(data, fs=fs)[0]
    ceps=np.array(ceps)
    for x in ceps:
        obs.append(np.array(x))
    vect.append(ceps)
    tmt.append(valu(fnm))

pickle.dump(obs, open( "obs.pickle", "wb" ) )
pickle.dump(vect, open( "features.pickle", "wb" ) )
pickle.dump(tmt, open( "class.pickle", "wb" ) )

"""
obs = pickle.load( open( "obs.pickle", "rb" ) )
vect = pickle.load( open( "features.pickle", "rb" ) )
tmt = pickle.load( open( "class.pickle", "rb" ) )

obs=np.array(obs)

print len(obs),len(obs[0])

print "good"

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

accu={}
cns={}
paccu={}
pcns={}

for i in range(1,8):
    accu[i]=0
    cns[i]=0
    paccu[i]=0
    pcns[i]=0
    
print "good"

mina=maxa=len(codes[0])

for x in codes:
    if (len(x)<mina):
        mina=len(x)
    if(len(x)>maxa):
        maxa=len(x)

sz=len(tgt)

#X = np.column_stack([vq.whiten(codes), tgt])

X=np.array(codu)

tgt=np.array(tgt)

tr=X[0:((4*sz)/5)]
trg=tgt[0:((4*sz)/5)]
ts=X[((4*sz)/5):]
tsg=tgt[((4*sz)/5):]

n_classes=7

import pyaudio
import wave
import cv2

clf = neighbors.KNeighborsClassifier(7, weights='uniform')
    
for weights in ['uniform']:
    print "total size: ",len(tgt)

    clf.fit(tr,trg)

    pred = clf.predict(tr)

    for i in range(1,8):
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

    for i in range(1,8):
        print i,"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/(pcns[i]+0.001))
        print i,"recall : ",accu[i],cns[i],((accu[i]*(1.0))/(cns[i]+0.001))

    pred = clf.predict(ts)

    for i in range(1,8):
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

    for i in range(1,8):
        print antival(i),"precision : ",paccu[i],pcns[i],((paccu[i]*(1.0))/(pcns[i]+0.001))
        print antival(i),"recall : ",accu[i],cns[i],((accu[i]*(1.0))/(cns[i]+0.001))

    input()

    chunk=1024

    def getFileReady(name):
	p = pyaudio.PyAudio()
	stream = p.open(format = pyaudio.paInt16,channels = 1,	rate = 44100,input = True, frames_per_buffer = chunk)
	wf = wave.open(name+'.wav', 'wb')
	wf.setnchannels(1)
	wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
	wf.setframerate(44100)
	return (p,stream,wf)

    while True:
        responded=False
	(p,stream,wf)=getFileReady("temp")
        i=0
        print "rec"
	while i<200:
		data = stream.read(chunk)
		wf.writeframes(data)
		i+=1
	stream.close()
	p.terminate()
	wf.close()
	data, fs = wavread("temp.wav")[:2]
        ceps = mfcc(data, fs=fs)[0]
        ceps=np.array(ceps)
        ceps=vq.whiten(ceps)
        t=ceps
        l=len(ceps)
        print l
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

          print antival(clf.predict(kst))
