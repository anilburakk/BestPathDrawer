# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:17:19 2019

@author: Anıl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = pd.read_excel('Coordinates.xlsx', 'Sayfa1')
xy_coor = file.as_matrix()
coord = xy_coor

file2=pd.read_excel('distancematrix.xls','Sayfa1')
dist1=file2.as_matrix()
distances=[]

n=len (xy_coor)-1
for i in range(len (dist1)-1):
    dist1[i+1][i+2]=0
for i in range (len (dist1)-1):
    distances.append(dist1[i+1][2:])
Distances=np.asanyarray(distances)

def routes (n):
    cities = np.arange(n)
    
    np.random.shuffle(cities)
     
    cities = np.append(cities,cities[0])
    
 #In this part cities are created and added on the numpy array   
    return cities


def distance (ar):
    totalway = 0
    for i,j in zip(ar[:-1],ar[1:]):
        totalway = totalway + Distances[i][j]
    return totalway

def drawing_path(aroute):
    #In this part , drawn path for the first element of population matrix
    for i,j in zip(aroute[:-1],aroute[1:]):
        plt.plot([coord[i][1],coord[j][1]],[coord[i][0],coord[j][0]],'-o')
    plt.show()
    
    
def creatingbetterpath(path1, path2,r):
    #Genetic Algorithm
    path1 = path1[:-1]
    path2 = path2[:-1]
    mesafe=[]
    
    #In this part, Path 3 to 1 is created and connected.
    for i,j in zip(path1[:-1],path1[1:]):
        mesafe.append( Distances[i][j])
    s     = np.random.randint(80)    
    path3_1 = np.hstack((path1[:s], path2[s:]))
    unique, counts = np.unique(path3_1, return_counts=True)
    d = dict(zip(unique, counts))
    relocate=[]
    for i in d:
        if d[i]==2:
            relocate.append(i)
    if len(set(path3_1))!=len(set(path2)):
        missing = list(set(path1)-set(path3_1))
       
        for i,j in zip(relocate,missing):
            
            index=np.where(path3_1==i)[0][0]
                                       
            path3_1[index]=j
               
    #In this part, Path 3 to 2 is created and connected.        
    path3_2 = np.hstack((path1[:s], path2[s:]))
    unique, counts = np.unique(path3_2, return_counts=True)
    d = dict(zip(unique, counts))
    relocate=[]
    for i in d:
        if d[i]==2:
            relocate.append(i)
    if len(set(path3_2))!=len(set(path2)):
        missing = list(set(path1)-set(path3_2))
       
        for i,j in zip(relocate,missing):
            
            index=np.where(path3_2==i)[0][1]
                        
            path3_2[index]=j       
            
    if distance(path3_1)   <    distance(path3_2):
        path3=path3_1
        
    else:
        path3=path3_2
                
    a,b,c,d= np.random.randint(0,n-r, 4)
    for i,j in zip (np.arange(a,c+r),np.arange(b,d+r)):
        path3[i],path3[j] = path3[j], path3[i]
        
         
    path3 = np.append(path3,path3[0])
    return path3

def alternativebetterpart(path1,path2):
    path1 = path1[:-1]
    path2 = path2[:-1]
    s = np.random.randint(0,n)
    path3 = np.hstack((path1[:s],path2[s:]))
    unique, counts = np.unique(path3, return_counts=True)
    d = dict(zip(unique,counts))
    relocate=[]
    for i in d:
        if d[i]==2:
            relocate.append(i)
    if len(set(path3))!= len(set(path2)):
        missing = list(set(path1)-set(path3))
        for i,j in zip(relocate,missing):
            if np.random.rand()>.5:
                index = np.where(path3 == i)[0][0]
            else:
                index = np.where(path3==i)[0][1]
            path3[index] = j
    chance = np.random.rand()
    if chance>0.9:
        a,b = np.random.randint(0,n,2)
        path3[a],path3[b] = path3[b],path3[a]
    if chance>0.8 and chance <=0.9:
        a,b =np.random.randint(0,n-2,2)
        for i,j in zip(np.arrange(a,a+2),np.arrange(b,b+2)):
            path3[i],path3[j] = path3[j],path3[i]
    if chance>0.6 and chance <= 0.8:
        a,b =np.random.randint(0,n-3,2)
        for i,j in zip(np.arrange(a,a+3),np.arrange(b,b+3)):
            path3[i],path3[j] = path3[j],path3[i]
    if chance>0.3 and chance <=0.6:
        a,b =np.random.randint(0,n-5,2)
        for i,j in zip(np.arrange(a,a+5),np.arrange(b,b+5)):
            path3[i],path3[j] = path3[j],path3[i]
    if chance <=0.3:
        a,b =np.random.randint(0,n-8,2)
        for i,j in zip(np.arrange(a,a+8),np.arrange(b,b+8)):
            path3[i],path3[j] = path3[j],path3[i]
    path3 = np.append(path3,path3[0])
    return path3
def get_population_performance(population):
    # returns to total distances
    
    perf = []
    for i in population:
        perf.append(distance(i))
    return np.array(perf)

def listed_population(population):
    #listing and sorts the population according to totaldistances
    performance = get_population_performance(population)
    i = np.argsort(performance)
    return population[i]
def create_initial_population(n):
    #creates and sorts an initial population of size n by using listed_population.
    
    population = []
    length=81
    for i in range(n):
        p = routes(length)
        population.append(p)
    population=np.array(population)
    population = listed_population(population)    
    return population

"""
reproduces the best n individuals of a population to produce n*n new indivioduals and sorts the new population 
return the sorted new population
""" 
def circle(population,n,t):
  

    population=population[:n]
    newpopulation = []
    for i in population:
        for j in population:
            newpopulation.append(creatingbetterpath(i,j,t))
    newpopulation = np.array(newpopulation)
    newpopulation = listed_population(newpopulation)
    return newpopulation

n           = 81
population  = create_initial_population(500)
performances = []
population1=[]

for i in range(40):
   
    population = circle(population,25,1)
    performances.append(distance(population[0]))

    plt.plot(performances,'.-')
    plt.show()
    
    for t in range (10):
        population = circle(population,27,0)
        population = circle(population,27,1)
        population = circle(population,27,2)
        population = circle(population,27,3)
        population = circle(population,27,i+1)
        
        drawing_path(population[0])
        print('Iteration ', (10)*i+t+1, 'best total distance %5.2f'% distance(population[0]),'km.')


better_route= population[0]
better_route = np.delete(better_route,81)
for i,j in enumerate(better_route):
    if j==5:
        locationofankara = i
first = better_route[locationofankara:]
second = better_route[:locationofankara]
better_route = np.append(first,second)
better_route= np.append(better_route,5)    
print("The best route is ",better_route+1)

a = population[0]
a = np.delete(a,81)
for i in (a):
    if a[i]==5:
        b = i
half1 = a[b:]
half2 = a[:b]
a = np.hstack((half1,half2))

a = np.append(a,5)
print("Alternative Way",a+1)







