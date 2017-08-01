# encoding: utf-8
# cython: profile=True
# filename: chess_util.pyx

cimport cython
from cpython cimport array
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.table import Table
from libc.stdlib cimport rand, RAND_MAX

"""
a and b is a tuple (i,j) where i is the column number(index in the solution array) 
and j is the row number(value in the solution array) 
@note: attack function takes the coordinates of two queens and return a c array of two[attack or not, direction of attack]
"""
cdef int * attack(int *a,int *b):
    cdef int  result[2] 

    
    if a[0]==b[0]:
        raise ValueError("wrong value in column field")
   
    elif (a[1]==b[1]):
              
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
    
    
    cdef int i,j,attacks=0
    cdef int l = int(len(solution))
    cdef int * a
    cdef int * b
    a=<int *>PyMem_Malloc(2*cython.sizeof(int))
    if a is NULL:
        raise MemoryError()
    b=<int *>PyMem_Malloc(2*cython.sizeof(int))
    if b is NULL:
        raise MemoryError()
    for i in range(l-1):
        for j in range(i+1,l):

            a[0]=i
            a[1]=int(solution[i])
            b[0]=j
            b[1]=int(solution[j])
            attacks+=attack(a,b)[0]
    PyMem_Free(a)
    PyMem_Free(b)
    return attacks



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
    re =[0,0]
    c_re=attack(a,b)
    
    re[0]=c_re[0]
    re[1]=c_re[1]
   
    return re

# solution is a list of queens index in each column range from 1 to length of the list. 
def index_to_image(solution):
    N = len(solution)
    Z = solution
# G is a NxNx3 matrix
    G = np.zeros((N,N,3))

# Where we set the RGB for each pixel
    for i in range(N):
        for j in range(N):
            if ((i+j)%2)==0 :
                G[i][j] = [1,1,1]
        G[i][Z[i]-1] = [0,0,1]
    return G


"""
@author: yzhu
@param: data, an square matrix with the value of the evaluation result of a given solution, the corresponding element for queens will have a value of -1 

"""
def show_evaluation(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]
        if val<0:
            tb.add_cell(i, j, width, height, text='Q', 
                    loc='center', facecolor=color)
        elif val > 0:
            tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)
        else:
            tb.add_cell(i, j, width, height, 
                    loc='center', facecolor=color)

    # Row Labels...
    index =[i+1 for i in  range(nrows)]
    for i, label in enumerate(index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    columns = [i+1 for i in  range(ncols)]
    for j, label in enumerate(columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig


def test_visualization():
    solution = [3,4,2,3,5]
    print index_to_image(solution)
    #show_queens(solution)
    #data = pandas.DataFrame(np.random.random((5,5)))#,  columns=['A','B','C','D','E'])
    data = np.random.random((5,5)) 
    for j in range(len(solution)):
        data[solution[j]-1,j] = -1
    print data
    show_evaluation(data)

    plt.show()

"""
@note: return a random solution based on the input number of queens
"""
cpdef list random_solution(int n_queens = 8):
    cdef int i,tem 
    cdef list solution=[]
    for i in range(n_queens):
        tem = int((rand()/float(RAND_MAX)* (n_queens-1)))
        solution.append(tem)
    return solution



cpdef list get_sucessors(list solution): 
    cdef int i,j,l = len(solution)
    cdef list sucessors=[],new_solution
    for i in range(l):
        
        for j in range(l):
            new_solution = solution[:]
            if solution[i] != j: 
                new_solution[i] = j
                sucessors.append(new_solution)
            
    return sucessors