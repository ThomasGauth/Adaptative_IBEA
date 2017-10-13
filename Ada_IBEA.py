import os, sys
import math
from sys import exit
import time
import numpy as np
import cocoex
import random
from cocoex import Suite, Observer, log_level

def norma(val,b_max,b_min):
	# scale the value val between b_max and b_min
    return (val-b_min)/(b_max-b_min)

def fitness_eval(indice,size,F_norm,c,k):
	# compute "epsilon+" fitness values
    res = 0
    for i in range(size):
        if i!=indice :
            I = min([F_norm[i][0]-F_norm[indice][0],F_norm[i][1]-F_norm[indice][1]])
            res-=math.exp(-I/(k*c))
    return res

def suppr_ele(list,index):
	# remove the element at position index in a a list
    result = []
    for i in range(len(list)):
        if i != index:
            result += [list[i]]
    return result

def ada_ibea(fun, lbounds, ubounds, budget):
	# Algorithm implementation

    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)

    # Initialise parameters
    population_size = 50
    k=0.05
    mutation_prob = 0.05

    # Initial population of size population_size
    X = np.array(lbounds + (ubounds - lbounds) * np.random.rand(population_size, dim))

    # Iterate over budget
    while budget > 0:
    	# X image in the bi-objective space
        F = [fun(x) for x in X]
        size = len(X)
        F1 = [F[i][0] for i in range(size)]
        F2 = [F[i][1] for i in range(size)]
        b1_max,b1_min,b2_max,b2_min = max(F1),min(F1),max(F2),min(F2)
        # Scale each objective to the interval [0,1]
        F_norm = [[norma(F[i][0],b1_max,b1_min),norma(F[i][1],b2_max,b2_min)] for i in range(size)]
        # Compute indicator values and determine c
        c = 0
        for i in range(size):
            for j in range(size):
                eps = min([F_norm[i][0]-F_norm[j][0],F_norm[i][1]-F_norm[j][1]])
                if abs(eps)>c :
                    c=abs(eps)
        # Compute fitness values of individuals
        F_fitness = np.array([fitness_eval(i,size,F_norm,c,k) for i in range(size)])

        while len(F_fitness) >= population_size:

        	# Remove the individual with the smallest fitness value
            indice_suppr=0
            for i in range(1,len(F_fitness)):
                if F_fitness[i] < F_fitness[indice_suppr]:
                    indice_suppr = i

            X = suppr_ele(X,indice_suppr)
            F_fitness = suppr_ele(F_fitness,indice_suppr)

            # Update the fitness values of the remaining individuals
            for i in range(len(F_fitness)):
                I = min([F_norm[indice_suppr][0]-F_norm[i][0],F_norm[indice_suppr][1]-F_norm[i][1]])
                F_fitness[i] += math.exp(-I/(c*k))

        if budget <= 1 :
            # don't apply the genetic selection and return the best solution
            index_return = np.argmax(F_fitness)
            return X[index_return]

        #### Mating selection: binary tournaments
        # Done independently until alpha solutions are chosen 
        alpha = population_size//2
        while len(F_fitness) > alpha :
        	# uniformly chose two solutions at random from population with replacement
            fight = np.random.randint(0,len(F_fitness),2)
            # Tournament
            if F_fitness[fight[0]] > F_fitness[fight[1]] :
                X = suppr_ele(X,fight[1])
                F_fitness = suppr_ele(F_fitness,fight[1])
            else :
                X = suppr_ele(X,fight[0])
                F_fitness = suppr_ele(F_fitness,fight[0])


        #### Generic variations ( recombination + mutation )

        ## Recombination ( 25% of the new population)
        while len(X) < (alpha + (population_size - alpha)//2) :
        	# uniformly chose two solutions at random from population with replacement 
            couple = np.random.randint(0,len(X),2)
            # Randomly chose the intersection point between parents 
            point_intersection = np.random.randint(0,dim)
            # baby = crossover between parents
            baby = np.append(X[couple[0]][0:point_intersection], X[couple[1]][point_intersection:])   
            X += [baby]

        ## Mutations to complete X
        while len(X) < population_size :
        	# Chose an individual
            new = X[np.random.randint(0,len(X))]
            for i in range(len(new)) :
                lb, ub = lbounds[i], ubounds[i]
                # Apply mutation with a probability of mutation_prob
                if random.random()< mutation_prob :
                		new[i] = random.uniform(lb, ub)
            X += [new]

        ## Addition of 10 (random) new elements
        nb_random_add = 10
        for i in range(nb_random_add):
            new = X[0]
            for i in range(len(new)) :
                lb, ub = lbounds[i], ubounds[i]
                new[i] =  random.uniform(lb, ub)
		X += [new]


        budget -= 1
