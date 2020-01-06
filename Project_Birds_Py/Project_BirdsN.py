
# coding: utf-8

# # Proyecto Clasificación de aves Parte 1
# # Manejo de datos, Segmentación, y extracción de caracteristicas
# ### (La segmentacion es totalmente nuestra, basada en lareferencia [5] del informe)

# ### MANEJO DE LOS DATOS

# ### Dataset

# In[1]:


from scipy.io.wavfile import read, write
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import xml.etree.ElementTree as ET  #reading the labels xml file

folder=('D:/Training_Data/data/')  #file directory

def ExtractLabels(labels_dir):      #this rutine processes the xml file and returns an array with the info of a given file name
    tree=ET.parse(labels_dir)
    root=tree.getroot()
    labels=list()
    for i in range(0,len(root.getchildren())):      #iterates for the number of elements 
        labels.append(root[i].tag)                   #root.tag is "Audio", but root[].tag are the labels of the items, and root[].text its content
    for i in range(0,len(root.getchildren())):
        labels.append(root[i].text)
    labels=np.array([labels]).T
    labels=labels.reshape(len(root.getchildren()),2, order='F') #the array is a column, i reshape it into 2 columns
    return labels

def getSample(specie,labels):   #X
    import random
    li=list()
    for i in range(0,len(labels)):
        if(specie==labels[i][np.where(labels[i]=='Species')[0],1]):
            li.append(i)
    return random.choice(li)

def matchedSpecie(ns1,ns2,ns3,ns4,ns5,ns6,ns7,ns8,labels): #returns a list with the index of the samples that match certain specie
    #count=0
    listi=list()
    y=[]
    for i in range(0,len(labels)):  #checks for the samples of a certain specie
        g=np.where(labels[i]=='Genus')[0][0]#np.where(arg)[0] returns the row where the label is "Genus"
        a=np.where(labels[i]=='Species')[0][0]#np.where(arg)[0] returns the row where the label is "Specie"
        if((labels[i][g,1]+'-'+labels[i][a,1])==ns1 or (labels[i][g,1]+'-'+labels[i][a,1])==ns2 or
           (labels[i][g,1]+'-'+labels[i][a,1])==ns3 or (labels[i][g,1]+'-'+labels[i][a,1])==ns4 or
           (labels[i][g,1]+'-'+labels[i][a,1])==ns5 or (labels[i][g,1]+'-'+labels[i][a,1])==ns6 or
           (labels[i][g,1]+'-'+labels[i][a,1])==ns7 or (labels[i][g,1]+'-'+labels[i][a,1])==ns8):  
            #count=count+1 
            listi.append(1)
            
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns1):
                y.append(1)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns2):
                y.append(2)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns3):
                y.append(3)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns4):
                y.append(4)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns5):
                y.append(5)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns6):
                y.append(6)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns7):
                y.append(7)
            if(labels[i][g,1]+'-'+labels[i][a,1]==ns8):
                y.append(8)
            
        else:
            listi.append(0)
    return listi,y                   #indices with 1 means that this is a sample of one of the species chosen

def getlistlabel(labelst,genus,name,count):
    #this fuction returns a list with the names that appear on the data about a certain label country, species, etc
    va=[]
    
    for i in range(0,len(labelst)):
        #labelst is a list, each element is a table containing the labels, but
        #the amount of labels vari and sometimes specie may be the 18th label, other times 17th who knows, so i have to
        #get where is it in this specific file
        a=np.where(labelst[i]==name)[0][0]
        if(genus=='Genus'):
            #If genus is selected then the list takes in count the genus-specie 
            g=np.where(labelst[i]==genus)[0][0]
            
            va.append(labelst[i][g,1]+'-'+labelst[i][a,1])
        else:
            va.append(labelst[i][a,1])
        #labelst is a list of arrays, each array has 2 columns, the first is the title and the second is the actual value/name
        #i obtaned the index of the row that containes the specie, and the second column is the name of each specie, i got them
        #on a list
        #print(va)
    #but i want to get the uniques values, i need a numpy array
    va=np.array(va)
    listunique=np.unique(va)
    counts=[]
    if(count=='yes'):
        #this section counts how many elements with a certain name are in the label list
        for each in range(0,listunique.shape[0]):
            #gets the element name 
            n=0
            for i in range(0,len(va)):
                #counts how many times said element appears on the label list
                if(listunique[each]==va[i]):
                    n=n+1
            counts.append(n)
        listunique=listunique[:,None]
        counts=np.array(counts)[:,None]
        
        listunique=np.column_stack((listunique,counts))
    #if yes is selected then this function returns an array with the unique elements and how many of them are
    #if not, returns a 1D vector with the unique names
    return listunique


# ### El dataset es bastante grande, consiste en mas de 11k audios con mas de 500 especies, sin embargo trabajaremos con 8 especies

# ### Procesando el Dataset desde el directorio

# In[2]:


file=list()        #a list for the sound files
labels_dir=list()      #list for the xml files
b=os.listdir(folder)  #makes a list with the names of the files in the diretory
b=np.array(b)         #transform it to an array
r=0
for i in range(0,b.shape[0]):
    if(b[i].split('.')[1]==('wav')):   #if its a wav file concatenates it to file
        if(r==1):
            print(b[i])
        file.append(b[i])
        r=1
    elif(b[i].split('.')[1]==('xml')):
        if(r==2):
            print(b[i])                    # !!! there are 4 xml files without wav file. are useless
        else:
            labels_dir.append(b[i])           #if its an xml file concatenates it to file
        r=2
#the 4 .xml files without .wav are shown


# In[3]:


len(file),len(labels_dir) #There have to be the same amount of .wav and .xml


# ### Ya tengo los archivos de audio separados de los archivos etiquetas

# In[4]:


#Creating a list of the data on the .xml files in order to read them
labelst=list()
for i in range(0,len(file)):
    labelst.append(ExtractLabels(folder+labels_dir[i]))   #adds the array with the info of the audio to labels list
    print(i, end='\r')


# ### Aqui escogemos que 8 especies a trabajar. por tomar algunas sacamos las 8 con mas muestras

# In[5]:


#species=getlistlabel(labelst,'Nada','Country','yes')
#species

species=getlistlabel(labelst,'Genus','Species','yes') 
order=-np.int_(species[:,1])
species[np.argsort(order)][0:10,:]


# In[6]:


listi,y=matchedSpecie('Cercomacra-serva','Schistocichla-leucostigma','Drymophila-devillei','Grallaria-guatimalensis',
                      'Liosceles-thoracicus','Hylophilus-ochraceiceps','Lepidothrix-coronata','Ammodramus-aurifrons',labelst) 

y=np.array(y)[:,None]
print(y.shape)


# ### ya tenemos las muestras de interés, ahora digitalizamos los audios

# In[7]:


fr=list()      #list for the frequency rate of each audio signal
sam=list()     #list for the samples of each audio signal

labels=list()
ind=list()
lenfile=len(file)
for i in range(0,len(file)):
    if(listi[i]==1):
        print(round(100*i/lenfile,2), '%', end='\r')
        #ind.append(i)
        f ,s =read(folder+file[i])    #digitalizes the signal of an audio sample
        fr.append(f)                  #adds it to the list fr
        if s.ndim > 1:              #checks if the audio is mono or stereo
            s=s[:,0]
        sam.append(s)               #adds the signal to sam

        #with the function ExtractLabels i obtain an array with the info of an audio
        labels.append(ExtractLabels(folder+labels_dir[i]))   #adds the array with the info of the audio to labels list


# #### De los audios se extrajo la frecuencia de muestreo (fr), la señal muestreada (sam)
# #### Y las etiquetas de cada muestra de audio (labels)

# ## Observamos una muestra en concreto

# In[8]:


#a=ind[285]
#file[a]


# In[87]:


#labels[2]
p=getSample('aurifrons',labels)  #gets a random sample of a specie
p,y[p]


# In[88]:


labels[p]


# In[86]:


frep,timep, specp=signal.spectrogram(sam[p],fr[p])
#plt.pcolormesh(times[p], freqs[p][0:40], specs [p][0:40,:])
plt.pcolormesh(timep, frep, specp)
plt.title('Espectrograma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
specpdb=20*np.log10(specp)
plt.pcolormesh(timep, frep, specpdb)
plt.title('Espectrograma dB')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# In[77]:


x = sam[p]/float(np.max(abs(sam[p]))) # escala la amplitud de la senal
t = np.arange(0, float(len(x))/fr[p], 1.0/fr[p]) # Vector de tiempo
plt.figure()
plt.plot(t,x) # Dibujar la grafica
# Los siguientes dos comandos dibujan las etiquetas de los ejes
plt.xlabel('Time',fontsize=18) # Etiqueta eje X
plt.ylabel('Amplitude',fontsize=18) # Etiqueta eje Y
plt.show() # Mostrar la grafica
#Audio(x, rate=fr[p]) # para escuchar la senal, si se desea


# # Segmentación
# ### El algoritmo de segmentacion se hizo basado en la referencia [5] delinforme
# ### Esta se seencuentra en la carpeta "Segmentation"

# ### Antes de Extraer Caracteristicas, es necesario hacer una segmentacion y filtrado de los audios para cortar segmentos que no aportan informacion de la especie

# In[9]:


from Segmentation import SegmentS, butter_bandp

segments=list()

for i in range(0,len(sam)):
    print(i,end='\r' )
    segm=SegmentS(sam[i],fr[i],tdb=90,fdb=90,printprocess='NO',prefilter='NO',postfilter='NO')
    segments.append(segm)
    


# ### Observamos las duraciones de los audios antes y despues de la segmentación

# In[11]:


print('Longitudes')
print('i     Tiempo sin Segmentar       Tiempo segmentado')
for i in range(0,len(sam)):
    print(i,'   ',len(sam[i])/fr[i],'          ',len(segments[i])/fr[i])


# ### Comparando un dato Original con uno segmentado (La funcion Audio() puede tardar)

# In[17]:


#p=getSample('aurifrons',labels)  #gets a random sample of a specie
p=0
print(p,y[p])
ts = np.arange(0, float(len(sam[p]))/fr[p], 1.0/fr[p]) # Vector de tiempo
plt.figure()
plt.plot(ts,sam[p]) 

plt.xlabel('Time',fontsize=18) # Etiqueta eje X
plt.ylabel('Amplitude',fontsize=18) # Etiqueta eje Y
plt.show() # Mostrar la grafica
Audio(sam[p], rate=fr[p]) # para escuchar la senal, si se desea


# In[18]:


tse = np.arange(0, float(len(segments[p]))/fr[p], 1.0/fr[p]) # Vector de tiempo
if(len(tse)>len(segments[p])):
    tse=np.delete(tse,len(tse)-1,axis=0)
plt.figure()
plt.plot(tse,segments[p]) 

plt.xlabel('Time',fontsize=18) # Etiqueta eje X
plt.ylabel('Amplitude',fontsize=18) # Etiqueta eje Y
plt.show() # Mostrar la grafica
Audio(segments[p], rate=fr[p]) # para escuchar la senal, si se desea


# ### A continuacion partidmos los audios en segmentos de duracion similar, esto hace aumentar el numero de muestras de 507 a 3528

# In[54]:


nseg=list()
ny=np.empty(shape=(1,1))
nfr=list()
for i in range(0,len(sam)):
    tim=np.arange(0, float(len(segments[i])/fr[i]), 1.0/fr[i])
    tmax=tim[tim.shape[0]-1]
    if(tmax>=2):
        for j in range(0,int(tmax/2)):
            tinf=np.where(tim==2*j)[0][0]
            tsup=np.where(tim==2*(j+1))[0][0]
            #print( tinf,tsup, np.where(tmax==tim)[0][0])
            nseg.append(segments[i][tinf:tsup])
            ny=np.row_stack((ny,y[i]))
            nfr.append(fr[i])
        if(np.where(2*int(tmax/2)==tim)[0][0]!=tim.shape[0]-1):
            nseg.append(segments[i][np.where(2*int(tmax/2)==tim)[0][0]:tim.shape[0]-1])
            ny=np.row_stack((ny,y[i]))
            nfr.append(fr[i])
            #print( np.where(int(tmax/2)*2==tim)[0][0],tim.shape[0]-1, np.where(tmax==tim)[0][0])
    else:
        nseg.append(segments[i])
        ny=np.row_stack((ny,y[i]))
        nfr.append(fr[i])
ny=np.delete(ny,0,axis=0)


# In[55]:


print('Longitudes')
for i in range(0,len(nseg)):
    print(i,'   ',len(nseg[i])/nfr[i],'  ',nfr[i],'  ', ny[i],'   ',len(nseg[i]))


# # Extraccion de caracteristicas

# #### Las caracteristicas se extraen con MFCC (libreria extraida de https://github.com/jameslyons/python_speech_features/tree/master/python_speech_features)

# In[56]:


#MFCC eature extraction
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from scipy import stats
#mfcc_feat=list()
#d_mfcc_feat=list()
#fbank_feat=list()


# In[59]:


#mfcc_feat=np.zeros(shape=(1,52),dtype='float32')
mfcc_feat=np.zeros(shape=(1,52),dtype='float32')

dmfcc_feat=np.zeros(shape=(1,52),dtype='float32')
for i in range(0,len(nfr)):
    mf = mfcc(nseg[i],nfr[i],winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1200,
              lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    
    #delta MFCC optional
    #dmf=delta(mf,2)
    #dmfm=np.mean(dmf,axis=0)
    #dmfstd=np.std(dmf,axis=0)
    #dmfkur=stats.kurtosis(dmf,axis=0)
    #dmfskew=stats.skew(dmf,axis=0)
    #dmfcc_feat=np.row_stack((dmfcc_feat,np.column_stack((dmfm[:,None].T,dmfstd[:,None].T,dmfkur[:,None].T,dmfskew[:,None].T))))
    
    #Mfcc
    mfccm=np.mean(mf,axis=0)
    mfccstd=np.std(mf,axis=0)
    #mfccvar=np.var(mf,axis=0)
    mfcckur=stats.kurtosis(mf,axis=0)
    mfccskew=stats.skew(mf,axis=0)
    mfcc_feat=np.row_stack((mfcc_feat,np.column_stack((mfccm[:,None].T,mfccstd[:,None].T,
                                                       mfcckur[:,None].T,mfccskew[:,None].T))))
    print(mfcc_feat.shape, end='\r')
mfcc_feat=np.delete(mfcc_feat,0,axis=0)
dmfcc_feat=np.delete(dmfcc_feat,0,axis=0)


# In[60]:


#for i in range(0,mfcc_feat.shape[0]-1):
#    if(np.isnan(mfcc_feat[i,0])):
#        mfcc_feat=np.delete(mfcc_feat,43,axis=0) #why does this data is nan
#        y=np.delete(y,43,axis=0)
mfcc_feat.shape,dmfcc_feat.shape  #mfcc_feat matriz de caracteristicas para cada una de las muestras, y matriz de clases


# In[61]:


mfcc_feat.shape,dmfcc_feat.shape  #mfcc_feat matriz de caracteristicas para cada una de las muestras, y matriz de clases


# ## Para finalizar esta primera parte guardamos las caracteristicas en un archivo

# In[62]:


data=np.column_stack((mfcc_feat,ny))    #only mfcc ---- Best

#data=np.column_stack((mfcc_feat,dmfcc_feat))  #mfcc and delta
#data=np.column_stack((data,y))

#data=np.column_stack((dmfcc_feat,y))    #only delta
#concatenates each sample with the labels before shuffling so each sample is still paired with its label
np.savetxt('DSEG2SNF.txt',data,newline='\n')
data.shape

