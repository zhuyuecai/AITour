
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.table import Table

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

def show_queens(solution):
    G = index_to_image(solution)
    plt.imshow(G,interpolation='nearest')
    plt.show()


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
        else:
            tb.add_cell(i, j, width, height, text=fmt.format(val), 
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




if __name__== '__main__':
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
