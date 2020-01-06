
# coding: utf-8

# # Proyecto Clasificación de aves Parte 2
# # Clasificación con RNA 

# ### Importando los datos

# In[2]:


import numpy as np

data=np.loadtxt('DSEG2SNF.txt')


# ### Una primera prueba

# In[3]:


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


# In[19]:


from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(30,30,30),alpha=0.7,max_iter=800)
mlp.fit(dxt,dyt[:,0])  #Training  the NN      #trying less features

predicted=mlp.predict(dxte)[None].T     #trying less features
#predicted=mlp.predict(dxte)[None].T
acc=(np.sum(predicted==dyte))/(dyte.shape[0])
acc*100


# ## Definiendo la partición de los datos y la validacion cruzada

# In[4]:


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
                     
        
        
    


# In[5]:


indFolds=KFolds(dyt[:,0],10)#list which each element is the training and test folds


# ### Funcion Para hallar el mejor modelo y validacion

# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def CrossV(X,Y,Folds=10,nnshape=(30,30,30),maxiter=200,alpha=0.01,average='NO'):
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
        
        mlp=MLPClassifier(hidden_layer_sizes=nnshape,max_iter=maxiter,alpha=alpha)
        #creates the nn
        
        mlp.fit(xtrain,ytrain) #traines the nn
        tpredictions=mlp.predict(xtrain) #predicts the same data used to train to see if we are overtraiing
        
        Err_train=np.append(Err_train,np.sum(tpredictions!=ytrain)/tpredictions.shape[0])
        #Error in training
        F1_train=np.concatenate((F1_train,f1_score(ytrain,tpredictions,average=None)[None]),axis=0)
        #concatenates the f1 score of each iteration
        
        predictions=mlp.predict(xtest) #clasifies the test folder
        Err_test=np.append(Err_test,np.sum(predictions!=ytest)/predictions.shape[0])
        F1_test=np.concatenate((F1_test,f1_score(ytest,predictions,average=None)[None]),axis=0)
    if(average=='YES'):
        return np.mean(Err_train,axis=0),np.mean(Err_test,axis=0),np.mean(F1_train,axis=0),np.mean(F1_test,axis=0)
    else:
        return Err_train,Err_test,F1_train,F1_test
    
def BestMIter(X,Y,alpha,nnshape=(30,30,30),plot='YES'):
    scales=50
    niter=10
    Ets=1
    err=np.empty(shape=(0,))
    alps=np.empty(shape=(0,))
    for i in range(scales):
        iters=10+(i*5)
        Etr,Et,F1tr,F1t=CrossV(X,Y,Folds=10,nnshape=nnshape,maxiter=iters,alpha=alpha,average='YES')
        print(iters,Et)
        if(plot=='YES'):
            err=np.append(err,Et)
            alps=np.append(alps,iters)
        if(Et<Ets):
            Ets=Et
            niter=iters
def BestM(X,Y,maxiter=800,nnshape=(30,30,30),plot='YES'):
    scales=9
    nalpha=10**(-7)
    Ets=1
    err=np.empty(shape=(0,))
    alps=np.empty(shape=(0,))
    for i in range(scales):
        alpha=10**(i-7)
        Etr,Et,F1tr,F1t=CrossV(X,Y,Folds=10,nnshape=nnshape,maxiter=maxiter,alpha=alpha,average='YES')
        print(alpha,Et)
        if(plot=='YES'):
            err=np.append(err,Et)
            alps=np.append(alps,alpha)
        if(Et<Ets):
            Ets=Et
            nalpha=alpha
    if(plot=='YES'):
        import matplotlib.pyplot as plt
        plt.loglog(alps, err, basex=10)
        plt.grid(True)
        plt.xlabel('Alpha')
        plt.ylabel('Loss')
        plt.show()
   
    if(plot=='YES'):
        import matplotlib.pyplot as plt
        plt.loglog(alps, err, basex=10)
        plt.grid(True)
        plt.xlabel('Alpha')
        plt.ylabel('Loss')
        plt.show()
    print('Best in Range ',nalpha)
    alpha2=nalpha/2
    for i in range(0,10):
        alpha=alpha2+(4*i*alpha2*10**-1)
        Etr,Et,F1tr,F1t=CrossV(X,Y,Folds=10,nnshape=nnshape,maxiter=maxiter,alpha=alpha,average='YES')
        print(alpha,Et)
        if(Et<Ets):
            Ets=Et
            nalpha=alpha
    print('Best Alpha ',nalpha)
    return nalpha


# In[8]:


#Etr,Et,F1tr,F1t=CrossV(dxt,dyt[:,0],Folds=10,nnshape=(30,30,30),maxiter=750,alpha=1,average='YES')
#Etr,Et,F1tr,F1t


# ### Variando el alpha

# In[70]:


alpha=BestM(dx,dy[:,0])


# ### Con el mejor alpha Hallado realizamos validación cruzada y obtenemos los puntajes

# In[83]:


Etr,Et,F1tr,F1t=CrossV(dx,dy[:,0],Folds=10,nnshape=(30,30,30),maxiter=800,alpha=0.7,average='NO')


# In[84]:


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

# In[97]:


a=np.around(F1tr,3)
b=np.around(F1t,3)
np.savetxt('F1_TRAINING.txt',a,fmt='%.2f',newline='\n')
np.savetxt('F1_TEST.txt',b,fmt='%.2f',newline='\n')


# ### Variando las epocas (No converge hasta cerca de las 800 epocas)

# In[ ]:


alpha=BestMIter(dx,dy[:,0],0.7,nnshape=(30,30,30),plot='YES')


# In[65]:




