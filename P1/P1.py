#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import xlrd

file_location='DatosPunto1.xlsx'
Data=xlrd.open_workbook(file_location) #loading the excel file with xlrd library
sheet=Data.sheet_by_index(0)           #Data is not an array so i have to make an array manualy with each cell
dat_array=np.zeros((sheet.nrows-1,sheet.ncols-1), dtype=np.float64) #i create an array of zeroes because im really bad with python, also, the size is
                                                                    #1 column less because i wont use Y2, and 1 row less because the first row are the labels, X1, X2, etcc
for col in range(sheet.ncols-1):                           
	col=col
	for rows in range(sheet.nrows-1):
		rows=rows
		dat_array[rows,col]=sheet.cell_value(rows+1,col)  #asigning each value on a cell to the matrix but ignoring the first row because that is the lable

#i have my data loaded, now i have to make my neural network
np.random.shuffle(dat_array)

#splitting my data 85% for the CrossValidation and 15% for final test

tn=int(dat_array.shape[0]*0.80)

dat=dat_array[0:tn,0:8] #input data
out=dat_array[0:tn,8:9] #output answer
datft=dat_array[tn:dat_array.shape[0],0:8] #final test data, 15%(not test as cross validation)
outft=dat_array[tn:dat_array.shape[0],8:9]

#test data                                                    #i splitted the data 80% for training and 20% for the test
#dat_p=dat_array[tn:dat_array.shape[0],0:8]
#out_p=dat_array[tn:dat_array.shape[0],8:9]
#print(out_p[0])

d_norm = (dat-np.mean(dat,axis=0))/(np.std(dat,axis=0))  #normalizing data
#d_pnorm = (dat_p-np.mean(dat_p,axis=0))/(np.std(dat_p,axis=0))
dft_norm = (datft-np.mean(datft,axis=0))/(np.std(datft,axis=0))




def NNTrainingLR(dat, out, alpha, epo):
        
        #Defining weights

        w=np.random.rand(dat.shape[1]+1,1) #1 neuron only on the last layer, 8 input neurons +1 bias
        on=np.ones(shape=(dat.shape[0],1))       #this creates the bias for all samples
        #on=2*on
        a_1=np.column_stack((on,dat))     #concatenating the bias column
        for m in range(0,epo):
                J=0
                dJ=0
                a_2=np.dot(a_1,w)       #this is the same as h
                #J= J + (np.dot((a_2- Out).T,(a_2 - Out))/(2*a_2.shape[0]))
                dJ=(np.dot(((a_2- out).T),a_1)).T/(a_1.shape[0])
                w=w - alpha*(dJ)
                w[1:]=w[1:] - alpha*0.1/a_1.shape[0]*w[1:]
                #print(J)        
        
                                
        return [w]

def NN(dat, w):                        #this is the prediccion function
        a_1=np.insert(dat,0,1,axis=1)
        a_2=np.dot(a_1,w)
        return[a_2] 
         
def NNTrainingLMS(dat,out, alpha, epo):
        
        #Defining weights

        w=np.random.rand(dat.shape[1]+1,1) #1 neuron only on the last layer, 8 input neurons +1 bias
        on=np.ones(shape=(dat.shape[0],1))       #this creates the bias for all samples
        #on=2*on
        a_1=np.column_stack((on,dat))     #concatenating the bias column
        for m in range(0,epo):
                
                dJ=0
                a_2=np.dot(a_1,w)       #this is the same as h
                
                dJ=(np.dot(((a_2- out).T),a_1)).T
                w=w - alpha*(dJ)
                w[1:]=w[1:] - alpha*0.1/a_1.shape[0]*w[1:]
                #print(J)        
        
                                
        return [w]
def NNTest(dft_norm,outft,w):             #evaluates a data set for the test
        ecm=0
        [out_t]=NN(dft_norm,w)  #test the LR for one data
        ecm=np.dot(((out_t-outft).T),(out_t-outft))/(outft.shape[0])
        ecm=numfromarray(ecm)
        
        return ecm
def numfromarray(ar):
        ar=ar[0]
        ar=ar[0]
        return ar

#cross validation
def CrossV(d_norm,out,alpha,typ):
        tn=int(d_norm.shape[0]/5)
        r=d_norm.shape[0]%5
        #for i in range(5):
        d1=d_norm[0:tn,:]
        o1=out[0:tn,:]
        d2=d_norm[tn:2*tn,:]            #splitting the data in 5 folds for the cross validation
        o2=out[tn:2*tn,:]
        d3=d_norm[2*tn:3*tn,:]
        o3=out[2*tn:3*tn,:]
        d4=d_norm[3*tn:4*tn,:]
        o4=out[3*tn:4*tn,:]
        d5=d_norm[4*tn:5*tn+r,:]  #this last one has a bit more data because the division is not exact
        o5=out[4*tn:5*tn + r,:]
        ecm=[0, 0, 0, 0, 0]

        if typ==0:
                datc=np.concatenate((d2,d3,d4,d5))
                outc=np.concatenate((o2,o3,o4,o5))
                [w]=NNTrainingLR(datc,outc,alpha,100)
        else:
                datc=np.concatenate((d2,d3,d4,d5))
                outc=np.concatenate((o2,o3,o4,o5))
                [w]=NNTrainingLMS(datc,outc,alpha,10)
                
        ecm[0]=NNTest(d1,o1,w)
        
        if typ==0:
                datc=np.concatenate((d1,d3,d4,d5))
                outc=np.concatenate((o1,o3,o4,o5))
                [w]=NNTrainingLR(datc,outc,alpha,100)
        else:
                
                datc=np.concatenate((d1,d3,d4,d5))
                outc=np.concatenate((o1,o3,o4,o5))
                [w]=NNTrainingLMS(datc,outc,alpha,10)

        ecm[1]=NNTest(d2,o2,w)

        if typ==0:
                datc=np.concatenate((d1,d2,d4,d5))
                outc=np.concatenate((o1,o2,o4,o5))
                [w]=NNTrainingLR(datc,outc,alpha,100)
        else:
                
                datc=np.concatenate((d1,d2,d4,d5))
                outc=np.concatenate((o1,o2,o4,o5))
                [w]=NNTrainingLMS(datc,outc,alpha,10)
        ecm[2]=NNTest(d3,o3,w)

        if typ==0:
                datc=np.concatenate((d1,d2,d3,d5))
                outc=np.concatenate((o1,o2,o3,o5))
                [w]=NNTrainingLR(datc,outc,alpha,100)
        else:
                
                datc=np.concatenate((d1,d2,d3,d5))
                outc=np.concatenate((o1,o2,o3,o5))
                [w]=NNTrainingLMS(datc,outc,alpha,10)
        ecm[3]=NNTest(d4,o4,w)

        if typ==0:
                datc=np.concatenate((d1,d2,d3,d4))
                outc=np.concatenate((o1,o2,o3,o4))
                [w]=NNTrainingLR(datc,outc,alpha,100)
        else:
                
                datc=np.concatenate((d1,d2,d3,d4))
                outc=np.concatenate((o1,o2,o3,o4))
                [w]=NNTrainingLMS(datc,outc,alpha,10)
        ecm[4]=NNTest(d5,o5,w)

        ecm=(ecm[0] +ecm[1]+ecm[2]+ecm[3]+ecm[4])/5
        return [ecm]

def BestA(d_norm,out,typ):
        ecm=1000000
        e=0
        alpha=0
        if typ==0:
                print('LR')
        else:
                print('LMS')
        for i in range(30):
                if typ==0:
                    alp=4*(i+1)/100
                else:
                    alp=4*(i+1)/100000
                [e]=CrossV(d_norm,out,alp,typ)
                print('Alpha   ',alp,'ECM   ', e)              
                if e<ecm:
                        #print(alp)
                        alpha=alp
                        ecm=e
        return [alpha]


# ### Entrenando el LR y variando el alpha para encontrar el mas optimo en varios ordenes

# In[87]:


[alpha]=BestA(d_norm,out,0)
print('Best Alpha LR', alpha)


# In[88]:


[w]=NNTrainingLR(d_norm,out,alpha, 100)
ecmlr=NNTest(dft_norm,outft,w)
ecmlr


# ### Entrenando el LMS y variando el alpha para encontrar el mas optimo en varios ordenes

# In[89]:


[alpha]=BestA(d_norm,out,1)
print('Best Alpha LMS', alpha)


# In[90]:


[w]=NNTrainingLMS(d_norm[0:520,:],out[0:520,:],alpha, 20)
ecmlms=NNTest(dft_norm,outft,w)
ecmlms


# ### Entrenar con demasiados datos el LMS hace que el algoritmo diverja, por lo que fué necesario limitar la cantidad de datos para entrenarlo

# In[ ]:




