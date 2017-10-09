import os, sys
import math
from sys import exit
import time
import numpy as np
import cocoex
import random
from cocoex import Suite, Observer, log_level

def norma(val,b_max,b_min):
    return (val-b_min)/(b_max-b_min)

def fitness_eval(indice,size,F_norm,c,k):
    res = 0
    for i in range(size):
        if i!=indice :
            I = min([F_norm[i][0]-F_norm[indice][0],F_norm[i][1]-F_norm[indice][1]])
            res-=math.exp(-I/(k*c))
    return res

def suppr_ele(list,index):
    result = []
    for i in range(len(list)):
        if i != index:
            result += [list[i]]
    return result

def ada_ibea(fun, lbounds, ubounds, budget):
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    population_size = 50
    k=0.05
    mutation = 0.1
    X = np.array(lbounds + (ubounds - lbounds) * np.random.rand(population_size, dim))
    while budget > 0:
        F = [fun(x) for x in X]
        size = len(X)
        F1 = [F[i][0] for i in range(size)]
        F2 = [F[i][1] for i in range(size)]
        b1_max,b1_min,b2_max,b2_min = max(F1),min(F1),max(F2),min(F2)
        F_norm = [[norma(F[i][0],b1_max,b1_min),norma(F[i][1],b2_max,b2_min)] for i in range(size)]
        c = 0
        for i in range(size):
            for j in range(size):
                # esp <0 ???
                eps = min([F_norm[i][0]-F_norm[j][0],F_norm[i][1]-F_norm[j][1]])
                if abs(eps)>c :
                    c=abs(eps)

        F_fitness = np.array([fitness_eval(i,size,F_norm,c,k) for i in range(size)])

        while len(F_fitness) >= population_size:
            indice_suppr=0
            for i in range(1,len(F_fitness)):
                if F_fitness[i] < F_fitness[indice_suppr]:
                    indice_suppr = i

            X = suppr_ele(X,indice_suppr)
            F_fitness = suppr_ele(F_fitness,indice_suppr)

            for i in range(len(F_fitness)):
                I = min([F_norm[indice_suppr][0]-F_norm[i][0],F_norm[indice_suppr][1]-F_norm[i][1]])
                F_fitness[i] += math.exp(-I/(c*k))

        if budget <= 1 :
            ##### je ne sais pas trop
            index_return = np.argmax(F_fitness)
            return X[index_return]

        #### partie selection genetique ( je garde les 50% meilleurs )
        while len(F_fitness) > (population_size//2) :
            fight = np.random.randint(0,len(F_fitness),2)
            if F_fitness[fight[0]] > F_fitness[fight[1]] :
                X = suppr_ele(X,fight[1])
                F_fitness = suppr_ele(F_fitness,fight[1])
            else :
                X = suppr_ele(X,fight[0])
                F_fitness = suppr_ele(F_fitness,fight[0])


        #### partie variation genetique ( croisement + mutation )
        ## croisement ( 25% de la nouvelle pop)
        while len(X) < ((3*population_size)//4) :
            couple = np.random.randint(0,len(X),2)
            baby = []
            for i in range(dim):
                baby +=[(X[couple[0]][i]-X[couple[1]][i])/2]

            X += [baby]

        ## mutation
        while len(X) < population_size :
            new = X[np.random.randint(0,len(X))]
            for i in range(len(new)) :
                lb, ub = lbounds[i], ubounds[i]
                new[i] = min([max([new[i] + mutation*random.uniform(lb, ub),lb]),ub])
            X += [new]

        ## ajout random (pas fait)
        nb_random_add = 10
        for i in range(nb_random_add):
            new = X[0]
            for i in range(len(new)) :
                lb, ub = lbounds[i], ubounds[i]
                new[i] =  random.uniform(lb, ub)
		X += [new]


        budget -= 1
