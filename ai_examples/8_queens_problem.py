import sys
from matplotlib.pyplot import xlabel, ylabel
sys.path.append('../')
import chess_util
import algorithms  
import matplotlib.pyplot as plt
import numpy as np

hill_climb = algorithms.Hill_Climbing()
solution = hill_climb.hill_climbing(chess_util.random_solution,chess_util.get_sucessors,chess_util.count_all_attacks,2)


data = np.zeros((8,8)) 
for j in range(len(solution)):
    data[solution[j],j] = -1

chess_util.show_evaluation(data)

print hill_climb.cost_history
plt.show()

plt.plot(hill_climb.cost_history)
plt.ylabel('cost')
plt.xlabel('trials')

plt.show()