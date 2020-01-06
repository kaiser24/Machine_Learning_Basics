#!/usr/bin/env python
# coding: utf-8

# # Tarea 3. Punto 1

# ### Definiciones Iniciales
# ### La funcion fitness Calcula el beneficio total sobre el peso total de una solucion pero retorna cero cuando el peso spbrepasa la capacidad maxima
# ### El Criterio de apareamiento se emplea recombinacion uniforme tomando un elemento aleatoriamente de cada padre
# ### Para el Criterio de mutacion se toman n bits aleatorios y se invierten

# In[20]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def F1(crom,maxC,Mat):
    #MAt would be the matrix with the values of the benefits (column0) and weights (column1) for each item (10 items)
    #maxC is the maximum weight the bag can carry
    #crom is the matrix with the cromosomes
    
    R=Mat[:,1,None]/Mat[:,0,None]
    #Obtaines the relation benefit/weight
    fitness=np.dot(crom,R)
    #fitness function
    
    #Calculates the total weight of each cromosome
    wt=np.dot(crom,Mat[:,0,None])
    #if the weight exceeds the maximum value, then that cromosome is useless
    for i in range(0,wt.shape[0]):
        if(wt[i]>maxC):
            fitness[i]=0
    
    return fitness
#this function is supose to be the fitness function

def UniformR(P1,P2):
    #this is the pairing criteria which is taking randomly a component of one of the parents
    P=P1
    P=np.row_stack((P,P2))
    M=np.random.randint(0,2,(1,10))
    M=np.row_stack((M,(M)))
    C=M*P + (1-M)*(np.flip(P,axis=0))
    return C
def nBitFlip(P,n):
    #this function flips n bits 
    c=np.array(P)    #####PYTHON DOESN'T CREATE NEW OBJECTS WITH ASSIGNMENTS???!!!!!!!!
    for i in range(0,n):
        f=np.random.randint(0,10,1)
        #print(f)
        if(c[f]==1):
            c[f]=0
        else:
            c[f]=1
    return c
        
    


# ### Datos

# In[21]:


Mat=np.array([[1.2,0.8,1.5,1.6,1,0.7,1.2,0.6,1.4,1,0.25,0.5,0.5,0.8,0.9,0.4,0.2,0.15,0.4,0.6]]) #peso y beneficio de los items
Mat=np.reshape(Mat,(10,2),order='F')
print(np.array([['Peso','Beneficio']]))
print(Mat)


# In[22]:


Tm=0.4;           #Tasa de mutacion 
Tc=0.4;          #Tasa de cruce  Porcentaje de parejas van a ser seleccionadas para el cruce
population=20; #Tamaño de la poblacion
N=20;           #Numero de generaciones
n=2;        #Number of bits to flip
maxC=5;     #peso maximo de la mochila Kg


# In[23]:


Pop=np.random.randint(0,2,size=(population,10)) #Matix of random 1's and 0's
for gen in range(N):  
    # Fitness evaluation
    Fitness = F1(Pop,maxC,Mat).T[0] 
    #obtains a matrix containing the fitness for each chromosome
     
    BestIndividuos=np.argsort(-Fitness,axis=0) #returns a vector which arguments are the indices of the cromosomes from best t worst fitness
    Pop = Pop[BestIndividuos]  #Reorganizes the population matrix from best to worst as says the previous obtained vector
  
    Wheel = np.cumsum(Fitness[BestIndividuos])/np.sum(Fitness[BestIndividuos])#creates a cummulative vector of the fitness and re-arange to 0-1  
    #cumsum returns the cummulative sum on a given axis, if no axis is given it does it along all elements, sum just sums all elements

    # Crossover - Wheel 
    
    for parent in range(int(Tc*population/2)): 
        #Tc is the rate of parenting, 0.4*pop/2 =4 -> 8 childs
        P = np.random.rand(1)  #creates a random number, if Wheel exceeds this number, take that parent
        ParentSelected_1 = np.where(Wheel >= P)[0][0]
        
        P = np.random.rand(1)
        ParentSelected_2 = np.where(Wheel >= P)[0][0]
        
        #Crossover Uniform Recombination
        Childs=UniformR(Pop[ParentSelected_1],Pop[ParentSelected_2])
        #Add to population
        #print(Childs)
        Pop = np.row_stack((Pop,Childs))
    
    
    # Mutation - Wheel
    for parent in range(int(Tm*population)):
        #Tm is the ocurrence rate of a mutation, so, with 0.4*pop =8, it tries 8 times to get a child with a mutation
        P = np.random.rand(1)
        ParentSelected = np.where(Wheel >= P)[0][0]
        
        #Add to population
        Child=nBitFlip(Pop[ParentSelected],n)
        #criteria for mutation is n bit flip
        Pop = np.row_stack((Pop,Child[None,:]))
    #Selection
    Fitness = F1(Pop,maxC,Mat).T[0] #Evaluates the fitness with the new members added
    BestIndividuos = np.argsort(-Fitness) #Sorts from best to worst
    
    Pop = Pop[BestIndividuos[:population]] #only takes the best 10 and discard the rest
    
    plt.plot(gen,F1(Pop[0,None],maxC,Mat)[0,0],'.')
    #this shows only the fitness of the best cromosome through the N iterations
plt.xlabel('Mejor Cromosoma a lo largo de Generaciones')
plt.ylabel('Fitness')
plt.show()


# In[24]:


print('Mejor Solucion')
print(Pop[0,None])


# In[ ]:





# In[ ]:




