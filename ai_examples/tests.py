import unittest
import numpy as np
import sys
sys.path.append('../')
#from chess_visualization import index_to_image
#import pyximport
#pyximport.install()

import chess_util

class Test_visualization(unittest.TestCase):
    def setUp(self):
        pass
 
    def test_integer(self):
        solution = [1,2,4,5,2]
        G = np.array([[[ 0,  0,  1],
  [ 0,  0,  0],
  [ 1,  1,  1],
  [ 0,  0,  0],
  [ 1,  1,  1]],

 [[ 0,  0,  0],
  [ 0,  0,  1],
  [ 0,  0,  0],
  [ 1,  1,  1],
  [ 0,  0,  0]],

 [[ 1,  1,  1],
  [ 0,  0,  0],
  [ 1,  1,  1],
  [ 0,  0,  1],
  [ 1,  1,  1]],

 [[ 0,  0,  0],
  [ 1,  1,  1],
  [ 0,  0,  0],
  [ 1,  1,  1],
  [ 0,  0,  1]],

 [[ 1,  1,  1],
  [ 0,  0,  1],
  [ 1,  1,  1],
  [ 0,  0,  0],
  [ 1,  1,  1]]])
        self.assertEqual( (chess_util.index_to_image(solution)- G).all(), False)
 

class Test_util(unittest.TestCase):
    def setUp(self):
        pass
    def test_attack(self):
        self.assertEqual( chess_util.pattack([3,4],[4,4]), [1,1]) 
        self.assertEqual( chess_util.pattack([3,4],[4,5]), [1,2]) 
        self.assertEqual( chess_util.pattack([3,4],[2,5]), [1,3])
        self.assertEqual( chess_util.pattack([3,4],[2,4]), [1,4])
        self.assertEqual( chess_util.pattack([3,4],[2,3]), [1,5]) 
        self.assertEqual( chess_util.pattack([3,4],[4,3]), [1,6])  
        self.assertEqual( chess_util.pattack([3,4],[4,6]), [0,0]) 
    def test_count_all_attacks(self):
         self.assertEqual( chess_util.count_all_attacks([3,4,2,3,5]), 3)



if __name__ == '__main__':
    unittest.main()
