#!/usr/bin/env python
# coding: utf-8

# # Red neuronal trabajando el dataset como una matriz completa

# In[1]:


import numpy as np

nchar=76

file_loc='DataNN.txt'
text=open(file_loc,'r') #loads the file as a text type
num_lines = sum(1 for line in open(file_loc)) #counts how many lines the text has
dat_array=np.zeros((num_lines-1,nchar), dtype=np.float64)
out_array=np.zeros((num_lines-1,1), dtype=np.int)




def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))



buf=text.readline()        #each time this function is called it automatically increases the pointer for the line, and i dont know how to navigate it
                           #so i call it 1 time to ignore the first line of the text which contains the names of the descriptors

for i in range(0,num_lines-1):    #takes the data from the text into an array
    buf=text.readline()         #calls for a line
    buf=buf.split(',')          #splits said line into a string of objects
    buf=np.array([buf])         #converts the string into an array
    dat_array[i,:]=buf[0,0:nchar]  #takes 20 characteristics from 76
    out_array[i,0]=buf[0,76:77] #the last object is \n from the text, the 76th object is the class answer

#dat_array=np.delete(dat_array,10,axis=1) #deleting too small characteristics

data=np.column_stack((dat_array,out_array)) #concatenating the descriptors with the outputs for shuffling
#a=np.array([range(0,data.shape[0])])
#a=a.T
#data=np.column_stack((data,a))


np.random.shuffle(data)               #shuffling data and reassigning 
dat_array=data[:,0:nchar]                #concatenating is important so the descriptors vector doesn´t loose its class
out_array=data[:,nchar:nchar+1]
out_array=out_array.astype(int)

dat_norm = (dat_array-np.mean(dat_array,axis=0))/(np.std(dat_array,axis=0))  #normalizing data Z score
#dat_norm = (dat_norm-np.min(dat_norm,axis=0))/(np.max(dat_norm,axis=0)-np.min(dat_norm,axis=0))     #min max

tn=int(dat_norm.shape[0]*0.85)

d_norm=dat_norm[0:tn,:] #splitting data 85% for Training and validation
out=out_array[0:tn,:]

dft_norm=dat_norm[tn:dat_norm.shape[0],:] #15% for the final test
outft=out_array[tn:dat_norm.shape[0],:]


def NNSTraining(d_norm,out,alpha,epo,neurons):
    #defining wights
    w1=np.random.rand(neurons,d_norm.shape[1]+1) 

    w2=np.random.rand(neurons,neurons+1)

    w3=np.random.rand(12,neurons+1)
    #out=out[:,0]
    o=np.zeros(shape=(1,12))
    for i in range(0,out.shape[0]):
        v=np.zeros(shape=(1,12))
        v[0,out[i]-1]=1
        o=np.row_stack((o,v))
    #print(o)  
    o=np.delete(o,0,axis=0)
    for j in range(0,epo):
        a1=d_norm
        a1=np.insert(a1,0,0.5,axis=1)

        z1=np.dot(a1,w1.T)
            #z1=z1.round(3)
        a2=sigmoid(z1)
       
        a2=np.insert(a2,0,0.5,axis=1)
        z2=np.dot(a2,w2.T)
            #z2=z2.round(3)
        a3=sigmoid(z2)
        a3=np.insert(a3,0,0.5,axis=1)
        z3=np.dot(a3,w3.T)
            #z3=z3.round(3)
        a4=sigmoid(z3)
         #print(a4)
            
            #now the backpropagation for upgrading the weights
        #J = np.sum((o-a4)*(o-a4)) #the cost function
            
            #first, i have to define the deltas, which are the efect of the weights of a layer upon the next layer
            #the deltas of the last layer are the derivate of the cost function respect to each weight
        d4=(o-a4)*d_sigmoid(z3)

        d3=np.dot(d4,w3)
        d3=np.delete(d3,0,axis=1)
        d3=d3*d_sigmoid(z2)

        d2=np.dot(d3,w2)
        d2=np.delete(d2,0,axis=1)
        d2=d2*d_sigmoid(z1)

         #my doubs with this because of the size of a and d   
        dw3 = -np.dot(d4.T,a3) 
                                         
        dw2 = -np.dot(d3.T,a2) 
        dw1 = -np.dot(d2.T,a1) 

            
        w3 = w3 - alpha*dw3   #just applies the changes
        w2 = w2 - alpha*dw2
        w1 = w1 - alpha*dw1
        #print(J/2)
     
    return [w1,w2,w3]

def NNS(d_norm,w1,w2,w3):
    a1=d_norm
    a1=np.insert(a1,0,0.5,axis=1)
    z1=np.dot(a1,w1.T)
            #z1=z1.round(3)
    a2=sigmoid(z1)
    a2=np.insert(a2,0,0.5,axis=1)
    z2=np.dot(a2,w2.T)
            #z2=z2.round(3)
    a3=sigmoid(z2)
    a3=np.insert(a3,0,0.5,axis=1)
    z3=np.dot(a3,w3.T)
            #z3=z3.round(3)
    a4=sigmoid(z3)
    return [a4]

def NNSTest(dft_norm,outft,w1,w2,w3):
    #clas=list()
    [out_t]=NNS(dft_norm,w1,w2,w3)
    
    ot=np.argmax(out_t,axis=1)+1 #stacks the output for each sample
    ot=ot[:,None]

    accuracy=(np.sum(ot==outft))/(outft.shape[0]) #this is a conditional sum from the np library

    return accuracy

#cross validation
def CrossV(d_norm,out,alpha,neurons):
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
        acc=[0, 0, 0, 0, 0]

        #Training and test for every combination of the folds and obtaining the accuracy for each one
        datc=np.concatenate((d2,d3,d4,d5))
        outc=np.concatenate((o2,o3,o4,o5))
        [w1,w2,w3]=NNSTraining(datc,outc,alpha,1000,neurons)                
        acc[0]=NNSTest(d1,o1,w1,w2,w3)
        
        datc=np.concatenate((d1,d3,d4,d5))
        outc=np.concatenate((o1,o3,o4,o5))
        [w1,w2,w3]=NNSTraining(datc,outc,alpha,1000,neurons)
        acc[1]=NNSTest(d2,o2,w1,w2,w3)

        datc=np.concatenate((d1,d2,d4,d5))
        outc=np.concatenate((o1,o2,o4,o5))
        [w1,w2,w3]=NNSTraining(datc,outc,alpha,1000,neurons)
        acc[2]=NNSTest(d3,o3,w1,w2,w3)

        datc=np.concatenate((d1,d2,d3,d5))
        outc=np.concatenate((o1,o2,o3,o5))
        [w1,w2,w3]=NNSTraining(datc,outc,alpha,1000,neurons)
        acc[3]=NNSTest(d4,o4,w1,w2,w3)

        datc=np.concatenate((d1,d2,d3,d4))
        outc=np.concatenate((o1,o2,o3,o4))
        [w1,w2,w3]=NNSTraining(datc,outc,alpha,1000,neurons)
        acc[4]=NNSTest(d5,o5,w1,w2,w3)

        acc=(acc[0] +acc[1]+acc[2]+acc[3]+acc[4])/5    #for this alpha and number of neurons i average the accuracy
        return [acc]
def BS(d_norm,out,alpha,neurons):
    tn=int(d_norm.shape[0]*0.80)
    db_norm=d_norm[0:tn,:] #splitting data 80% for Training and validation
    outb=out[0:tn,:]
    dv_norm=d_norm[tn:d_norm.shape[0],:] #20% for validation
    outv=out[tn:d_norm.shape[0],:]
    [w1b,w2b,w3b]=NNSTraining(db_norm,outb,alpha,1000,neurons)
    return [NNSTest(dv_norm,outv,w1b,w2b,w3b)] 
    
def BestA(d_norm,out,neurons,cb):                      #finding the best alpha with crossV for each one, really slow
        acc=0
        alpha=0
        for i in range(2,5):                    
            for m in range(0,3):
                alp=0.00001*(m+1)*np.power(10,i)
                if cb==0:
                    [accuracy]=CrossV(d_norm,out,alp,neurons)
                else:
                    [accuracy]=BS(d_norm,out,alp,neurons)
                print('Alpha ',alp,'Acc ', accuracy)              
                if accuracy>acc:
                        #print(alp)
                        alpha=alp
                        acc=accuracy
        return [alpha,acc]
def BestNN(d_norm,out,cb):               #This evalueates crossV for each combination of neurons and alpha, and for each
    neurons=2                         #crossV it trains 5 times so this takes a LOT of time
    acc=0
    for i in range(4,11):
        print(i*2,' Neuronas')
        [alpha,accuracy]=BestA(d_norm,out,i*2,cb)
        if accuracy>acc:
            acc=accuracy
            alpha=alpha
            neurons=i*2
    return [neurons,alpha]


# In[8]:


d_norm.shape,out.shape  #this is how data has to be put in 
out


# # Entreno la red con el mejor alpha que encontré y la cantidad de neuronas en la capa oculta con el mejor desempeño

# In[2]:


[w1,w2,w3]=NNSTraining(d_norm,out,0.01,4000,8)
NNSTest(dft_norm,outft,w1,w2,w3)*100


# # Aciertos y fallos de la prueba

# In[3]:


print(' Acierto O','Error X')
for i in range(0,dft_norm.shape[0]):
    if (np.argmax(NNS(dft_norm[i:i+1],w1,w2,w3))+1)==outft[i]:
        c='O'
    else:
        c='X'
    print('Prediccion ', np.argmax(NNS(dft_norm[i:i+1],w1,w2,w3))+1,'Clase ',outft[i],'  ',c)


# ## Con este algoritmo hallo el mejor alpha con la mejor cantidad de neuronas pero debido a que emplea Bootstraping para cada uno

# #### Para la validacion itero 1000 veces para ahorrar tiempo al hallar tiempo, pero luego obtengo el numero de iteraciones que mejoran el comportamiento

# In[132]:


[neurons,alpha]=BestNN(d_norm,out,1)
print('Mejor Modelo ',neurons, 'Neuronas y ',alpha, 'Alpha')


# ## Con este algoritmo hallo el mejor alpha con la mejor cantidad de neuronas pero debido a que emplea CrossV para cada uno tarda demasiado tiempo

# In[139]:


#[neurons,alpha]=BestNN(d_norm,out,0)
#print('Mejor Modelo ',neurons, 'Neuronas y ',alpha, 'Alpha')


# In[ ]:




