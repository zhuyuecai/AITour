# encoding: utf-8
# cython: profile=True
# filename: chess_util.pyx
#from numpy cimport ndarray
cimport cython
from cpython cimport array
#import numpy as np
#cimport numpy as np

"""a and b is a tuple (i,j) where i is the column number(index in the solution array) and j is the row number(value in the solution array) """
cdef list attack(int *a,int *b):
    cdef list result =[0,0]
    
    if a[0]==b[0]:
        raise ValueError("wrong value in column field")
    if a[1]==b[1]:
         
        result[0] = 1
        if a[0] > b[0] :
            result[1] = 4
        else:
            #result.append(4)
            result[1] = 1
    elif b[0]-a[0] == b[1]-a[1] :
        #result.append(1)
        result[0] = 1
        if b[0]-a[0] > 0 :
            #result.append(2)
            result[1] = 2
        else:
            #result.append(5)
            result[1] = 5
    elif b[0]-a[0] == a[1]-b[1] :
        #result.append(1)
        result[0] = 1
        if b[0]-a[0] > 0 :
            #result.append(6)
            result[1] = 6
        else:
            #result.append(3)
            result[1] = 3
    else:
        #result.append(0)
        #result.append(0)
        result[0] = 0
        result[1] = 0
    
    return result

"""
@note: count_all_attacks take an solution as arguments and return the total number of possible attacks without considering if there is a blocking queen in the attack path 

"""
cpdef int count_all_attacks(list solution):
    



#--------------python wrappers-------------------------
#pattack() for unittest only
def pattack(aa,bb):
    cdef int a[2]
    a[0]=aa[0]
    a[1]=aa[1]
    #a = [2,3]
    cdef int b[2]
    b[0]=bb[0]
    b[1]=bb[1]
    return attack(a,b)

