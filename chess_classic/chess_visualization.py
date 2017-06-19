import matplotlib.pyplot as plt
import numpy as np

# Z is your data set
def show_queens(n_col,solution):
    N = n_col
    Z = solution
# G is a NxNx3 matrix
    G = np.zeros((N,N,3))

# Where we set the RGB for each pixel
    for i in range(N):
        for j in range(N):
            if ((i+j)%2)==0 :
                G[i][j] = [1,1,1]
        G[i][Z[i]] = [0,0,1]

    plt.imshow(G,interpolation='nearest')
    plt.show()
