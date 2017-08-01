# encoding: utf-8
# cython: profile=True, boundscheck=False, wraparound=False, nonecheck=False
# filename: hill_climbing.pyx

from cython.parallel import prange
import numpy as np
#from Cython.Runtime.refnanny import __cinit__
#from Cython.Utility.MemoryView import __dealloc__
cimport numpy as np
from numpy cimport ndarray

from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from processing import Pool
#from libc.math cimport floor

cdef class Hill_Climbing: 
    # the list used to store the cost history for algorithm analysis
    cdef public list cost_history 

    def __init__(self): 
        self.cost_history = []

    """
    hill climbing: 1-randomly generate a solution candidate(initial search node), 2-evaluate the generated solution, 3-based on the given solution, 
    generate all possible search nodes(solution candidates) and evaluate them, 4-pick the one with the best evaluation result, 
    if multiple nodes map to the best evaluation result, randomly pick one, and repeat 2 to 4 until a perfect solution is found.
    If the search reach the shoulder, then randomly pick the other one with the best result. If after shoulder_steps, the solution is not found
    then stop and try again from step 1.
    If the next search nodes cannot give a better evaluation cost (ridge), then stop and try again from step1
    After max_trial tries, if no solution is found, then raise (no solution exception)
    
     
    """
    
    """
    private functions, return a bint indicating whether the first argument is better than the second
    """ 
    cdef bint __compare_cost__(self,float cost1,float cost2):
        return cost1 <= cost2
    
    cdef bint __compare_benefit__(self,float cost1,float cost2):
        return cost1 >= cost2
    
    cdef np.ndarray select_mins(self,np.ndarray x): 
        
        return np.where(x == x.min())[0]
    
    cdef np.ndarray select_maxs(self,np.ndarray x): 
        return np.where(x == x.max())[0]
    
    
    """
    @param initial_search_node: a function used to initialize the search node and return it
    @param  all_solution: a function used to generate all proceeding search nodes
    @param evaluate: a function used to evaluate the given search nodes and return the evaluation result, the correct solution should return 0  
    @param max_trial: the maximum number of trials
    @param shoulder_steps: maximum number of trials when the search reach the shoulder namely, the best evaluation result is the same as in the previous step 
    """
    cpdef hill_climbing(self,initial_search_node,get_sucessors,evaluate,int n_core,int max_trial=100, int shoulder_steps=10, objective_funtion = 'cost'):
        cdef int trials = 0 ,shoulder_trials = 0
        cdef int i,l,r
        cdef float cost
        cdef list sucessors
        #-------------------------------------
        #set up for parallel processing
        
        n_core = 2
        p = Pool(n_core)
        #-------------------------------------
        if objective_funtion == 'cost':
            select_best =self.select_mins
            compare_result = self.__compare_cost__
        elif objective_funtion == 'benefit':
            select_best = self.select_maxs
            compare_result = self.__compare_benefit__
        else:
            raise ValueError("Your objective_function should only be 'cost' or 'benefit!'")
        while (trials < max_trial):
            trials+=1
            current_node = initial_search_node()
            cost=evaluate(current_node)
            self.cost_history.append(cost) 
            while(True):
                
                if  cost == 0 :
                    return current_node
                elif cost < 0:
                    raise ValueError("Your evaluation give out a negative cost for solution %s"%(str(current_node)))
                else:
                    sucessors = get_sucessors(current_node)
                    l = len(sucessors)
                    results = np.zeros(l,dtype=float)
                    jobs = p.map(evaluate,sucessors)
                    #jobs = [evaluate(sucessors[i]) for i in range(l)]
                    
                    result = np.array(jobs)
                    #for i in prange(l,nogil = True):
                    #    result[i] = evaluate(sucessors[i])
                    bests = select_best(self,result)
    
                    #check if the best result is no better than the the current cost
                    if not compare_result(self,result[bests[0]] , cost  ):
                        break 
                    elif result[bests[0]] == cost:
                        if shoulder_trials > shoulder_steps: break
                        shoulder_trials +=1
                        if bests.size == 1:
                            current_node = sucessors[bests[0]]
                            
                        else: 
                            r = int((rand()/float(RAND_MAX)* (bests.size-1)))
                            current_node = sucessors[bests[r]]
                        self.cost_history.append(cost)
                    else:
                        shoulder_trials = 0
                        if bests.size == 1:
                            current_node = sucessors[bests[0]]
                            cost = result[bests[0]]
                        else:
                            r = int((rand()/float(RAND_MAX)* (bests.size-1)))
                            current_node = sucessors[bests[r]]
                            cost = result[bests[r]] 
                        self.cost_history.append(cost)
                
        return None   
            
            