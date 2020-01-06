#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
filelabel=('D:/U de A/Inteligencia Computacional/tarea4/')
dat=pd.read_csv(filelabel+'DatosAll.csv',header=None, sep=',').as_matrix()


# In[22]:


def kmeans(X,Y,nclusters,iteraciones=100,tol = 1e-4):
    sumprev=np.inf
    randomcenters=np.random.randint(X.shape[0],size=(nclusters))
    #takes the index of n samples randomly to use them to initialize the centers
    centers=X[randomcenters,:]
    #with the indices initializes the centers with the value of n samples picked randomly

    for j in range(iteraciones):
        distances=np.empty(shape=(X.shape[0],0))
        #a matrix with the distances of each sampleto each cluster, Nxc ,N samples x c clusters
        for i in range(nclusters):
            #distance_to_cluster = np.diagonal(np.sqrt(np.dot((X-centers[i]),(X-centers[i]).T)))
            distance_to_cluster = np.linalg.norm(X-centers[i],axis=1)
            #calculates the distance of each sample to the i-th center
            distances= np.column_stack((distances,distance_to_cluster[:,None]))
            #distances = np.concatenate((distances,distance_to_cluster[:,None]),axis=1)

        assign=np.argmin(distances, axis=1)
        #the index of the column that has the lowest value for each row,which is the lowest distance
        #fo each sample
        
        suma=0
        #now we have to recalculate the centers with the new members of each cluster
        for i in range(nclusters):
            centers[i]=np.mean(X[assign==i], axis=0)
            #the average value of each feature of the elememts that belong to the i-th cluster (lowest distance)
            
            #distance_to_cluster=np.sum(np.diagonal(np.sqrt(np.dot((X[assign==i]-centers[i]),(X[assign==i]-centers[i]).T))))
            distance_to_cluster = np.sum(np.linalg.norm(X[assign==i]-centers[i],axis=1))
            #recalculate the distance of the samples that belong to cluster i to its center and sums them
            
            suma= suma+distance_to_cluster
            #sums the distances of the samples to their clusters
        if np.abs(suma-sumprev)<tol:
            break;
            
        print(suma)
        
        sumprev=suma
        
    return assign   


# In[34]:


distance=np.diagonal(np.sqrt(np.dot((a-b),(a-b).T)))[:,None]


# In[24]:


np.linalg.norm((a-b),axis=1)


# In[ ]:




