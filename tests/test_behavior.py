import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(f'{str(Path(__file__).parent.absolute())}/../social_navigation_task')))
import preprocess as preprc
import numpy as np
import random

class TestBehavior(unittest.TestCase):
    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''
    compute = preprc.ComputeBehavior(file_path=None)

    def test_00_add_3rd_dimension_output_shape(self):
        for _ in range(10):
            U = np.random.randint(1,10, size=(np.random.randint(1,10),2)) # all the outputs should have this shape
            V = np.random.randint(1,10, size=(2,))
            ori = np.array([[0,0]])   
            U, V, ori = self.compute.add_3rd_dimension(U, V, ori)
            self.assertEqual(U.shape, V.shape, f'Adding in 3rd dimension made mismatched vector shapes {U.shape} != {V.shape}')  

    def test_01_cumulative_mean(self):
        rand_decs = np.array([random.choice([-1,1]) for _ in range(10)]).reshape(-2,2)
        resp_mask = rand_decs != 0
        cum_mean, (cum_sum, cum_count) = self.compute.cumulative_mean(rand_decs, resp_mask)
        self.assertEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], 'Cumulative mean for 1st dimension (ie, affiliation) is off')
        self.assertEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], 'Cumulative mean for 2nd dimension (ie, power) is off')

    def test_02_weight_decisions_constant(self):
        rand_decs  = np.array([random.choice([-1,1]) for _ in range(10)]).reshape(-2,2)
        rand_decs2 = self.compute.weight_decisions(rand_decs, decay='constant')
        self.assertListEqual(rand_decs.tolist(), rand_decs2.tolist(), 'The constant weighting is off: raw decisions should equal constant weighted (weight=1) decisions')
        
    # ori & U should have z-axis coordinates that increase sequentially; V should have a fixed z-axis coordinate

if __name__ == '__main__':
    unittest.main()