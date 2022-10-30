import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(f'{str(Path(__file__).parent.absolute())}/../social_navigation_analysis')))
import preprocess as preprc
import info 
import numpy as np
import pandas as pd
import random

from test_utils import *

# want to make meaningful assertions about the state after calling method 

class TestBehavior(unittest.TestCase):
    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''
    # TODO: add shape, type, etc testing for inputs
    # TDOD: test other methods - eg compute_character,compute_acrosscharacters 

    iters    = 5
    n_trials = 12
    compute = preprc.ComputeBehavior2(file=None)   

    #------------------
    # test input/output
    #------------------
    
    def test_xx_check_input(self):
        exp_shapes = [(2,2), (3,2)]
        to_check   = np.array([(3,2),[3,2]])

        bool = self.compute.check_input(to_check, exp_shapes)
        self.assertEqual(True, bool, 'Shape checker is off')

    def test_xx_run_output_shape(self):
        data    = random_behavior_data()
        compute = preprc.ComputeBehavior(file=data)
        compute.run()
        self.assertEqual(compute.behavior.shape[0], 63, 'There are not 63 rows in the outputted dataframe')
        # self.assertEqual(self.compute.behavior.shape[1], 15*num_models, f'There are not {} columns in the outputted dataframe')

    def test_get_decisions(self):
        data = random_behavior_data()
        decisions_raw = data[['affil', 'power']].values
        decisions = compute.get_decisions(decisions_raw, which='current')
        self.assertListEqual(decisions_raw.tolist(), decisions.tolist(), 'Current decisions off')

if __name__ == '__main__':
    unittest.main()