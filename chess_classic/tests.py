import unittest
import numpy as np
from chess_visualization import index_to_image

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
        self.assertEqual( (index_to_image(solution)- G).all(), False)
 





if __name__ == '__main__':
    unittest.main()
