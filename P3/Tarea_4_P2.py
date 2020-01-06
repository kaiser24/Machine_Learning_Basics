#!/usr/bin/env python
# coding: utf-8

# # Clasificación empleando C-means

# ### Cargo los datos

# In[521]:


import pandas as pd
import numpy as np
filelabel=('D:/U de A/Inteligencia Computacional/tarea4/')
datos=pd.read_csv(filelabel+'DatosAll.csv',header=None, sep=',').as_matrix()
clases=np.unique(dat[:,dat.shape[1]-1]).astype(int)
clases=clases-1
print(clases)


# In[522]:


#np.random.shuffle(data)
dx=datos[:,0:datos.shape[1]-1]
dy=datos[:,datos.shape[1]-1:datos.shape[1]]
dx= (dx-np.mean(dx,axis=0))/(np.std(dx,axis=0))  #normalizing data Z score


# ### Creo una clase para el objeto clasificador c-means

# In[523]:


class cmeans():
    def __init__(self, nclusters):
        self.nclusters=nclusters
        
    #training method
    def fit(self,X,iteraciones=100,tol = 1e-4,m=3):
        sumprev=np.inf
        #initializing randomly thematrix of membership
        numbers=np.random.randint(0,100,size=(X.shape[0],self.nclusters))
        u=np.zeros(shape=(X.shape[0],self.nclusters))
        for i in range(u.shape[0]):
            u[i]=numbers[i]/np.sum(numbers[i])
        
        self.centers=np.zeros(shape=(self.nclusters,X.shape[1]))
        
        for j in range(iteraciones):
            u2=u**m
            distances=np.empty(shape=(X.shape[0],0))
            #a matrix with the distances of each sampleto each cluster, Nxc ,N samples x c clusters
            for i in range(self.nclusters):
                self.centers[i]=np.dot(u2[:,i][None],X)/np.sum(u2[:,i])
                #with the membership calculates the centers of the clusters
                
                #the distances to the clusters
                distance_to_cluster = np.linalg.norm(X-self.centers[i],axis=1)
                #calculates the distance of each sample to the i-th center
                distances= np.column_stack((distances,distance_to_cluster[:,None]))
                #distances = np.concatenate((distances,distance_to_cluster[:,None]),axis=1)
            
            distances2=(1/distances)**(2/(m-1))
            distancesSum=np.sum(distances2,axis=1)[:,None]
            
            u=np.round((distances2/distancesSum),5)
            #recalculate the membership with the new distances to the clusters
            

            suma= np.sum(distances*u)
            #sums the distances of the samples to their clusters multiplied by the membership factor
            #print(suma, end='\r')
            if np.abs(suma-sumprev)<tol:
                break;
                
            sumprev=suma
       
        self.membership=u+1
        self.membership=self.membership-1
    def predict(self,X,m=3):
        distances=np.empty(shape=(X.shape[0],0))
        for i in range(self.nclusters):
                distance_to_cluster = np.linalg.norm(X-self.centers[i],axis=1)
                #calculates the distance of each sample to the i-th center
                distances= np.column_stack((distances,distance_to_cluster[:,None]))
                #distances = np.concatenate((distances,distance_to_cluster[:,None]),axis=1)
        distances2=(1/distances)**(2/(m-1))
        distancesSum=np.sum(distances2,axis=1)[:,None]
            
        u=np.round((distances2/distancesSum),5)
        #recalculate the membership with the new distances to the clusters   
        
        return u
    def correct(self, Y):#X    didnt work
        #reorganizes the centers as the labels say
        Y=Y.astype(int)-1
        uniques=np.unique(Y)
        reorg=np.zeros(shape=(0,2))
        
        self.prob_label=np.argmax(self.membership,axis=1)
        for i in uniques:
            realtag=np.bincount(Y[self.prob_label==i]).argmax()
            
            reorg=np.row_stack((reorg,np.array([i,realtag])))
            
        nassign=[]
        
        for i in range(reorg.shape[0]):
            nassign.append(np.where(reorg[:,1]==i)[0])
            
        nassign=np.array(nassign)
        self.centers=self.centers[nassign]


# In[524]:


def selectFeatures(X,selection):
    #this function cuts the data only taking the features selected
    #selection must be a string with the features separated by a space
    
    
    selection=selection.split(' ')
    selects=np.zeros(len(selection))#+labels
    for i in range(len(selection)):
        selects[i]=int(selection[i])
    selects=selects.astype(int)
    
    newData=np.zeros(shape=(X.shape[0],0))
    
    #now cutting the dataonly taking the selected features
    newData=X[:,selects]
    return newData


# In[525]:


sel=[[0],[0],[0],[0],[0],[0]]
sel[0]='2410 3881'
sel[1]='2410 3881 4018 3874 3946'
sel[2]='2410 3881 4018 3874 3946 4020 3882 3880 4019 3950'
sel[3]='2410 3881 4018 3874 3946 4020 3882 3880 4019 3950 3942 2430 3866 3792 3879 3944 4032 4030 4033 3956'
sel[4]='2410 3881 4018 3874 3946 4020 3882 3880 4019 3950 3942 2430 3866 3792 3879 3944 4032 4030 4033 3956 3876 1701 3943 3870 1533 1654 3958 3726 1890 4026 3940 4031 3868 3802 4028 2130 2202 3941 1873 1718 3878 3864 2145 2221 2203 1870 1887 4016 4021 3954'
sel[5]='2410 3881 4018 3874 3946 4020 3882 3880 4019 3950 3942 2430 3866 3792 3879 3944 4032 4030 4033 3956 3876 1701 3943 3870 1533 1654 3958 3726 1890 4026 3940 4031 3868 3802 4028 2130 2202 3941 1873 1718 3878 3864 2145 2221 2203 1870 1887 4016 4021 3954 2405 2333 1485 4034 2223 3805 4014 3863 3952 2331 2403 2199 2127 3862 3945 2183 4027 4029 2428 3804 1880 1889 3646 2348 2424 2201 2129 1721 2347 2423 2144 2220 3872 1553 3869 3951 1484 1535 4022 4017 3957 3867 4024 2404 2332 3875 3651 3953 2226 2222'
data1=selectFeatures(dx,sel[0])
datas=[]
for i in range(len(sel)):
    datas.append(selectFeatures(dx,sel[i]))


# ### Grafico laprueba para las 2 caracteristicas mejores, pues estás se pueden visualizar en un espacio bidimensional

# In[526]:


import matplotlib.pyplot as plt
for i in clases:
    dlabels=np.where(dy[:,0]==i)[0]
    plt.plot(data1[dlabels,0], data1[dlabels,1], 'x')

plt.xlabel('Agrupamiento original')
plt.show()


# ### Silhoulette Score de los datos con la clasificacion original
# 

# In[527]:


from sklearn.metrics import silhouette_score

silhouette_score(data1,dy[:,0]-1, metric='sqeuclidean')


# #### Clasifico Con las 2 mejores caracteristicas y grafico

# In[528]:


cm=cmeans(nclusters=clases.shape[0])
cm.fit(datas[0],m=3)


# ## Matriz de pertenencias de los datos a los clusters

# In[531]:


membership=cm.predict(datas[0],m=3)
np.round(membership,1)


# # C-Means entrega una matriz con un valor de pertenencia de cada dato a cada cluster, sin embargo, para graficar los datos tomo la asignacion del valor maximo de lapertenencia para cada muestra 

# In[532]:


assign=np.argmax(membership,axis=1)
for i in clases:
    dlabels=np.where(assign==i)[0]
    plt.plot(datas[0][dlabels,0], datas[0][dlabels,1], 'x')

plt.xlabel('Agrupamiento')
plt.show()


# ### Su score

# In[533]:


from sklearn.metrics import silhouette_score

silhouette_score(datas[0],assign, metric='sqeuclidean')


# ### Defino la funcion para validaion cruzada

# In[534]:


def KFolds(y, folds=5):
    uniques= np.unique(y) #creates a vector with the unique elements 
    indfolds=[np.empty(0,)]*folds
    globaladd=0
    
    for j in range(len(uniques)):
        datafoldj=np.where(uniques[j]==y)[0] #gets a vector with the indices of the samples that match this label
        count=np.sum(uniques[j]==y) #counts how many of samples of label j are
        val=int(count/folds) #how many of elements with label j for each fold
        
        randord=np.random.permutation(count)
        datafoldj=datafoldj[randord]
        
        for i in range(folds):
            #distribute samples with tag j evenly for each fold
            indfolds[i]=np.concatenate((indfolds[i],datafoldj[i*val:(i+1)*val]),axis=0)
        
        #how many samples were not distributed beacuse the division count/folds was not exact
        lack=count-val*folds
        for i in range(lack):#!!
            indfolds[globaladd]=np.append(indfolds[globaladd],datafoldj[lack+i])
            globaladd=globaladd+1
            
            if(globaladd==5):
                globaladd=0
    out=[]
    for i in range(folds):
        out.append([np.concatenate((np.array(indfolds)[np.where(np.arange(folds)!=i)[0]])).astype(int),indfolds[i].astype(int)])
        #reorganizes the folds so each element of out is one for test and the rest for training
    return out
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def CrossV(X,Y,Folds=5,nclusters=5,m=2,average='NO'):
    F1_train=np.empty(shape=(0,nclusters))
    F1_test=np.empty(shape=(0,nclusters))
    #Err_train=np.empty(shape=(0,))
    #Err_test=np.empty(shape=(0,))
    sil_train=np.empty(shape=(0,))
    sil_test=np.empty(shape=(0,))
    
    indFolds=KFolds(Y, Folds)
    #Gets the indices for the folds
    for i in range(Folds):
        xtrain,xtest=X[indFolds[i][0]],X[indFolds[i][1]]
        ytrain,ytest=Y[indFolds[i][0]],Y[indFolds[i][1]]

        scaler=StandardScaler()
        scaler.fit(xtrain)
        #obtaines the mean and std 
        
        xtrain=scaler.transform(xtrain)
        #standirizes the data
        
        cme=cmeans(nclusters=nclusters)
        #creates the nn
        
        cme.fit(xtrain,m=m) #traines the nn
        tpredictions=cme.predict(xtrain) #predicts the same data used to train to see if we are overtraiing
        tassign=np.argmax(tpredictions,axis=1)
        #Err_train=np.append(Err_train,np.sum(tpredictions!=ytrain)/tpredictions.shape[0])
        #Error in training
        
        sil_train=np.append(sil_train,silhouette_score(xtrain,tassign, metric='sqeuclidean'))
        #F1_train=np.concatenate((F1_train,f1_score(ytrain,tpredictions,average=None)[None]),axis=0)
        #concatenates the f1 score of each iteration
        
        predictions=cme.predict(xtest) #clasifies the test folder
        assign=np.argmax(predictions,axis=1)
        
        sil_test=np.append(sil_test,silhouette_score(xtest,assign, metric='sqeuclidean'))
        #Err_test=np.append(Err_test,np.sum(predictions!=ytest)/predictions.shape[0])
        #F1_test=np.concatenate((F1_test,f1_score(ytest,predictions,average=None)[None]),axis=0)
    if(average=='YES'):
        return np.mean(sil_train,axis=0),np.mean(sil_test,axis=0)
    else:
        return  sil_train,sil_test


# # Obtengo el S score para entrenamiento y prueba en validación para cada grupo delaseleccion de caracteristicas

# In[535]:


sil_tr=[]
sil_te=[]
for i in range(len(datas)):
    si,sit=CrossV(datas[i],dy,Folds=5,nclusters=5,m=1.1,average='NO')
    sil_tr.append(si)
    sil_te.append(sit)


# ## Grafico los resultados de las pruebas

# In[536]:


prueba=['2','5','10','20','50','100']
for j in range(len(datas)):
    folds=np.arange(1,6,1)
    plt.plot(folds,sil_tr[j], label='Silhouette Train '+prueba[j])
    plt.legend()
plt.show()
for j in range(len(datas)):
    folds=np.arange(1,6,1)
    plt.plot(folds,sil_te[j], label='Silhouette Test '+prueba[j])
    plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




