#Esta segmentacion esta basada en la referencia [5] del informe
 

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
        #dB, with 90dB it breaks into "words", with 30dB is easier to break, so it would segment into syllables
        #print(time[itn])
        #specdb[ifn,itn]  #checking how much dB does it fall when is not the signal


def SegmentS(signal, srate,tdb=90,fdb=90,printprocess='NO',prefilter='NO',postfilter='NO'):
    
    itn1=[]
    itn2=[]
    ifn1=[]
    ifn2=[]
    if(prefilter=='YES'):
        signal=butter_bandp(signal,500,10000,srate)
    t=np.arange(0,float(len(signal)/srate),1.0/srate)
    
    freqs, times, spec=spectrogram(signal,srate)
    specdb=20*np.log10(spec)
    
    tsize=spec.shape[1]
    fsize=spec.shape[0]
    ifn,itn=np.where(specdb==np.amax(specdb))[0][0],np.where(specdb==np.amax(specdb))[1][0]  
        #obtain the coordenates of the max value in dB
    pmax=specdb[ifn,itn]
    maxiter=int(t[t.shape[0]-1]*0.8)
    for f in range(0,maxiter):
        ifn,itn=np.where(specdb==np.amax(specdb))[0][0],np.where(specdb==np.amax(specdb))[1][0]  
        #obtain the coordenates of the max value in dB
        powermaxdB=specdb[ifn,itn]
        #gets the maximum power in the spectrogram
        if(powermaxdB<pmax-20):
            break
        
        #obtaining the section in time of a syllable
        #seccion   |----------!-----------------------------|
        #          0          max                           tsize
        #          |-----|----|------|----------------------|
        #          |   itn-i  itn  itn+i
        i=0
        for i in range(0,tsize-itn):
            if(specdb[ifn,itn+i]<(powermaxdB-tdb)):
            #when the power drops tdB then cut (forward)
                break
        itnv2=i+itn
        i=0
        for i in range(0,itn):
            if(specdb[ifn,itn-i]<(powermaxdB-tdb)):
            #when the power drops tdB then cut (backward)
                break
        itnv1=itn-i
        
        #saving the indices of the time for the secction
       
        itt1=np.abs(t-times[itnv1]).argmin()
        itt2=np.abs(t-times[itnv2]).argmin()
        itn1.append(itt1)
        itn2.append(itt2)

        if(postfilter=='YES'):
            #segmenting in frequency
            i=0
            for i in range(0,fsize-ifn):
                if(specdb[ifn+i,itn]<(powermaxdB-fdb)):
                    #when the power drops tdB then cut (Upper)
                    break
            ifnv2=i+ifn
            i=0
            for i in range(0,ifn):
                if(specdb[ifn-i,itn]<(powermaxdB-fdb)):
                    #when the power drops tdB then cut (Lower)
                    break
            ifnv1=ifn-i
            
            #saving the inces of the freq seccion
            ifn1.append(ifnv1)
            ifn2.append(ifnv2)
            
            prevsmax=specdb[ifn,itn]
            
            #print(freqs[ifnv1],freqs[ifnv2],times[itnv1],times[itnv2])
            specdb[ifnv1:ifnv2,itnv1:itnv2]=-np.inf
            if(printprocess=='YES'):
                #plt.pcolormesh(times,freqs,specdb)
                #plt.show()
                print(powermaxdB)
        else:
            prevsmax=specdb[ifn,itn]
            
            #print(freqs[ifnv1],freqs[ifnv2],times[itnv1],times[itnv2])
            specdb[:,itnv1:itnv2]=-np.inf
            if(printprocess=='YES'):
                #plt.pcolormesh(times,freqs,specdb)
                #plt.show()
                print(powermaxdB)

    if(postfilter=='YES'):
        it1=np.array(itn1)
        it2=np.array(itn2)
        if1=np.array(ifn1)
        if2=np.array(ifn2)
        it2=it2[np.argsort(itn1)]
        if1=if1[np.argsort(itn1)]
        if2=if2[np.argsort(itn1)]
        it1=it1[np.argsort(itn1)]
        ifh=int(np.mean(if2))
        ifl=int(np.mean(if1))
	    #filtering
        if(freqs[ifl]==0):
            signalfil=butter_bandp(signal, 500,freqs[ifh],srate)
        else:
            signalfil=butter_bandp(signal,freqs[ifl]+500,freqs[ifh]+1000,srate)  
        if(printprocess=='YES'):
            print('Filtering')
            f, ti, sp=spectrogram(signalfil,srate)
            sp=20*np.log10(sp)
            plt.pcolormesh(ti,f,sp)
    else:
        it1=np.array(itn1)
        it2=np.array(itn2)
        it2=it2[np.argsort(itn1)]
        it1=it1[np.argsort(itn1)]
        signalfil=signal

    count=0
    #segments
    segments=np.array([0])
    for i in range(0,it1.shape[0]):
        segments=np.concatenate((segments,signalfil[it1[i]:it2[i]]),axis=0)
        #print(segments.shape[0])
    
    
    return segments

#filtering
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    f = 0.5 * fs
    low = lowcut / f
    high = highcut / f
    if(high>1):
        high=0.99
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandp(signal, lowcut, highcut, fs,order=5): #bandpass filter
    b, a = butter_bandpass(lowcut, highcut,fs, order)
    y = lfilter(b, a, signal)
    return y