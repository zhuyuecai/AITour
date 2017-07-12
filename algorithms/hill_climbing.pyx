
import unittest
import chess_util

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