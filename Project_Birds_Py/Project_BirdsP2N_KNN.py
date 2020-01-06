
# coding: utf-8

# # Proyecto Clasificación de aves Parte 3
# # Clasificación con KNN 

# ### Importando Datos

# In[58]:


import numpy as np

data=np.loadtxt('DSEG2SNF.txt')


# ### Una primera prueba

# In[59]:


#np.random.shuffle(data)
dx=data[:,0:data.shape[1]-1]
dy=data[:,data.shape[1]-1:data.shape[1]]
dx= (dx-np.mean(dx,axis=0))/(np.std(dx,axis=0))  #normalizing data Z score
#dat_norm = (dat_norm-np.min(dat_norm,axis=0))/(np.max(dat_norm,axis=0)-np.min(dat_norm,axis=0))     #min max
tn=int(dx.shape[0]*0.85)

dxt=dx[0:tn,:]
dyt=dy[0:tn,:].astype(int)
dxte=dx[tn:dx.shape[0],:]
dyte=dy[tn:dy.shape[0],:].astype(int)
dxt.shape,dyt.shape,dxte.shape,dyte.shape,dxt.shape[0]+dxte.shape[0]


# In[60]:


np.unique(dy)


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
KNear=KNeighborsClassifier(n_neighbors=1)


# In[62]:


KNear.fit(dxt,dyt[:,0])
predicted=KNear.predict(dxte)[None].T     #trying less features
#predicted=mlp.predict(dxte)[None].T
acc=(np.sum(predicted==dyte))/(dyte.shape[0])
acc*100


# ## Definiendo la partición de los datos y la validacion cruzada

# In[1]:


import numpy as np
def KFolds(y, folds=10):
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def CrossV(X,Y,Folds=10,neighbors=50,average='NO'):
    F1_train=np.empty(shape=(0,8))
    F1_test=np.empty(shape=(0,8))
    Err_train=np.empty(shape=(0,))
    Err_test=np.empty(shape=(0,))
    
    indFolds=KFolds(Y)
    #Gets the indices for the folds
    
    for i in range(Folds):
        xtrain,xtest=X[indFolds[i][0]],X[indFolds[i][1]]
        ytrain,ytest=Y[indFolds[i][0]],Y[indFolds[i][1]]
        
        scaler=StandardScaler()
        scaler.fit(xtrain)
        #obtaines the mean and std 
        
        xtrain=scaler.transform(xtrain)
        #standirizes the data
        
        knear=KNeighborsClassifier(n_neighbors=neighbors)
        #creates the nn
        
        knear.fit(xtrain,ytrain) #traines the nn
        tpredictions=knear.predict(xtrain) #predicts the same data used to train to see if we are overtraiing
        
        Err_train=np.append(Err_train,np.sum(tpredictions!=ytrain)/tpredictions.shape[0])
        #Error in training
        F1_train=np.concatenate((F1_train,f1_score(ytrain,tpredictions,average=None)[None]),axis=0)
        #concatenates the f1 score of each iteration
        
        predictions=knear.predict(xtest) #clasifies the test folder
        Err_test=np.append(Err_test,np.sum(predictions!=ytest)/predictions.shape[0])
        F1_test=np.concatenate((F1_test,f1_score(ytest,predictions,average=None)[None]),axis=0)
    if(average=='YES'):
        return np.mean(Err_train,axis=0),np.mean(Err_test,axis=0),np.mean(F1_train,axis=0),np.mean(F1_test,axis=0)
    else:
        return Err_train,Err_test,F1_train,F1_test


# ### Funcion Para hallar el mejor modelo

# In[64]:


def BestM(X,Y,plot='YES'):
    scales=20
    Ets=1
    err=np.empty(shape=(0,))
    alps=np.empty(shape=(0,))
    for i in range(1,scales):
        alpha=i
        Etr,Et,F1tr,F1t=CrossV(X,Y,Folds=10,neighbors=alpha,average='YES')
        print(alpha,Et)
        if(plot=='YES'):
            err=np.append(err,Et)
            alps=np.append(alps,alpha)
        if(Et<Ets):
            Ets=Et
            nalpha=alpha


# ### Variando los k vecinos

# In[65]:


BestM(dx,dy[:,0])


# ### Con el mejor modelo Obtenemos los puntajes delas clases

# In[54]:


Etr,Et,F1tr,F1t=CrossV(dx,dy[:,0],neighbors=1,average='NO')


# In[55]:


import matplotlib.pyplot as plt
folds=np.arange(0,10,1)
plt.plot(folds,Etr,'g--', label='Error Entrenamiento')
plt.plot(folds,Et, 'r-o',label='Error Prueba')
plt.legend()
plt.show()
for i in range(8):
    plt.plot(folds,F1tr[:,i], label='F1 Score Clase '+str(i))
    plt.legend()
plt.show()
for i in range(8):
    plt.plot(folds,F1t[:,i], label='F1 Score Clase '+str(i))
    plt.legend()
plt.show()


# #### Guardo los resultados en un archivo

# In[57]:


a=np.around(F1tr,3)
b=np.around(F1t,3)
np.savetxt('F1_TRAINING_KNN.txt',a,fmt='%.2f',newline='\n')
np.savetxt('F1_TEST_KNN.txt',b,fmt='%.2f',newline='\n')


# In[69]:


graf=np.array([0.11230205774721595,
0.16573863375555303,
0.15478853303660087,
0.16015403457026864,
0.17187031536480535,
0.1756691472828789,
0.17071291540987793,
0.17804930381411768,
0.1857327525839865,
0.18936215418526317,
0.18788769040986214,
0.19743499666374392,
0.20979297356236853,
0.20413634434167088,
0.2157073047821234,
0.22131299285755318,
0.225943740233932,
0.23328352847207587,
0.23762986892062127,])


# In[74]:


plt.plot(np.arange(1,20),graf)
plt.title('KNN')
plt.xlabel('Neighbors')
plt.ylabel('Error')
plt.show()

