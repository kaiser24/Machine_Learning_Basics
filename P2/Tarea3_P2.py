#!/usr/bin/env python
# coding: utf-8

# # Tarea 3. Punto 2

# ### Datos de inicio

# In[1204]:


import numpy as np
import matplotlib.pyplot as plt

nc=4     #number of clusters
ndata=30 #number of data per cluster
poblacion=100   #number of cromosomes to evaluate
N=400        #generations
Tc=0.6      #coupling rate
Tm=0.6      #mutation rate
n=int((nc*ndata)*0.5)        #number of genes to reset


# ### Definiendo los datos

# In[1205]:


mean=[]
for means in range(0,nc):
    #making groups
    mx=means+5
    my=means+10
    if(means<nc/2):
        m=[mx*5,10]
    else:
        m=[(mx-nc/2)*5,20]
    mean.append(m)
cov=[[0.5,0],[0,4]]
print(mean)


# In[1206]:


#concatenating the groups in 1 array of data
x1=np.empty(shape=(nc,ndata))
x2=np.empty(shape=(nc,ndata))
y=np.zeros(shape=(nc,ndata))
for means in range(0,nc):
    x1[means], x2[means] = np.random.multivariate_normal(mean[means], cov, ndata).T
    y[means]=y[means]+means
    plt.plot(x1[means], x2[means], 'x')
plt.xlabel('Agrupamiento Original')
plt.show
data=np.array([[0,0]])
out=np.array([[0]])

for means in range(0,nc):
    dat=np.column_stack((x1[means,None].T,x2[means,None].T))
    data=np.row_stack((data,dat))
    out=np.row_stack((out,y[means,None].T))
data=np.delete(data,0,axis=0)
out=np.delete(out,0,axis=0)

#now data contains all the samples, Nx2, N being the total number of 
#data which is nc*ndata (number of clusters and number o data per cluster)
#for example, with 3 clusters and 400 data per cluster we have 3*400 =1200 total of samples


# In[1207]:


plt.plot(data[:,0],data[:,1],'x')
plt.xlabel('Datos')
plt.show()


# ### Empezamos con el Algoritmo Genetico

# ### Se tiene la funcion de fitness que es la suma de las distancias de los datos al centro de sus clusters
# ### Para el criterio de apareamiento se emplea recombinacion uniforme tomando un elemento aleatoriamente de cada padre
# ### Para el criterio de mutacion se emplea Reseteo aleatorio, tomando n genes aleatoriamente y asignandole un cluster aleatorio

# In[1208]:


def FitnessC(data,crom,nc):
    fitness=list()
    for m in range(crom.shape[0]):
    #for each cromosome
        sumtot=0
        center=np.empty(shape=(nc,data.shape[1]))
        for i in range(nc):
        #for each cluster obtaines the sum of the distances of the members to its center
            center[i]=np.mean(data[crom[m,:]==i],axis=0)
            #obtaines the center of each cluster
            sumdis=np.sum(np.linalg.norm(data[crom[m,:]==i]-center[i],axis=1))
            #first, this obtaines a vector with the distances 
            #of the data with the label to a cluster i , to the center of its cluster
            #then summs all of these values to obtain the sum of the distances of the members of a cluster to its center
            sumtot=sumdis+sumtot
            
        fitness.append(sumtot)
    return np.array(fitness)

def UniformR(P1,P2):
    #this is the pairing criteria which is taking randomly a component of one of the parents
    P=P1
    P=np.row_stack((P,P2))
    M=np.random.randint(0,2,(1,P1.shape[0]))
    M=np.row_stack((M,(M)))
    C=M*P + (1-M)*(np.flip(P,axis=0))
    return C

def RandomR(P,nc,n):
    #this function resets n genes to values between 0 and nc 
    c=np.array(P)    
    for i in range(0,n):
        f=np.random.randint(0,P.shape[0],1)
        e=np.random.randint(0,nc,1)
        c[f]=e
    return c
    


# In[1209]:


pop=np.random.randint(0,nc,(poblacion,nc*ndata))
for gen in range(N):  
    # Fitness evaluation
    Fitness = FitnessC(data,pop,nc) 
    #obtains a matrix containing the fitness for each chromosome
     
    BestIndividuos=np.argsort(Fitness,axis=0) #returns a vector which arguments are the indices of the cromosomes from best t worst fitness
    pop = pop[BestIndividuos]  #Reorganizes the population matrix from best to worst as says the previous obtained vector
  
    Wheel = np.cumsum(Fitness[BestIndividuos])/np.sum(Fitness[BestIndividuos])#creates a cummulative vector of the fitness and re-arange to 0-1  
    #cumsum returns the cummulative sum on a given axis, if no axis is given it does it along all elements, sum just sums all elements

    # Crossover - Wheel 
    
    for parent in range(int(Tc*poblacion/2)): 
        #Tc is the rate of parenting, 0.4*population/2 =4 -> 8 childs
        P = np.random.rand(1)  #creates a random number, if Wheel exceeds this number, take that parent
        ParentSelected_1 = np.where(Wheel >= P)[0][0]
        
        P = np.random.rand(1)
        ParentSelected_2 = np.where(Wheel >= P)[0][0]
        
        #Crossover Uniform Recombination
        Childs=UniformR(pop[ParentSelected_1],pop[ParentSelected_2])
        #Add to population
        #print(Childs)
        pop = np.row_stack((pop,Childs))
    
    
    # Mutation - Wheel
    for parent in range(int(Tm*poblacion)):
        #Tm is the ocurrence rate of a mutation, so, with 0.4*pop =8, it tries 8 times to get a child with a mutation
        P = np.random.rand(1)
        ParentSelected = np.where(Wheel >= P)[0][0]
        
        #Add to population
        Child=RandomR(pop[ParentSelected],nc,n)
        #criteria for mutation is n bit flip
        pop = np.row_stack((pop,Child[None,:]))
    #Selection
    Fitness = FitnessC(data,pop,nc) #Evaluates the fitness with the new members added
    BestIndividuos = np.argsort(Fitness) #Sorts from best to worst
    Fitness=Fitness[BestIndividuos]
    pop = pop[BestIndividuos[:poblacion]] #only takes the best 10 and discard the rest
    #print(Fitness[0])  #fitness best cromosome each generation
    


# ## Resultado de la Clasificacion

# In[1210]:


assignment=pop[0,:]
for i in range(0,nc):
    plt.plot(data[assignment==i,0],data[assignment==i,1],'x')
plt.xlabel('Clasificacion con GA')
plt.show()
for means in range(0,nc):
    plt.plot(x1[means], x2[means], 'x')
plt.xlabel('Agrupamiento original')
plt.show()


# #### Este algoritmo tiene el problema que no tiene en cuenta ningun orden para agrupar los datos, asi que los datos quedan en grupos diferentes de forma correcta, pero la etiqueta de cada grupo es basicamente de forma aleatoria

# In[ ]:



  


# In[ ]:





# In[ ]:




