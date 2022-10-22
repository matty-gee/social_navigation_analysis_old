import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(f'{str(Path(__file__).parent.absolute())}/../social_navigation_task')))
import preprocess as preprc
import info 
import numpy as np
import random

class TestBehavior(unittest.TestCase):
    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''

    # methods to test
    # compute character
    # computer across characters 
    
    iters = 10
    compute = preprc.ComputeBehavior(file_path=None)

    # TODO replace this! dont depend on an outside file... generate fake behavior instead?
    subject_compute = preprc.ComputeBehavior(file_path=info.example_xlsx, out_dir=False)

    def get_random_decs(self, n_trials=5):
        return np.array([random.choice([-1,1]) for _ in range(n_trials*2)]).reshape(-2,2)

    def test_xx_run_output_shape(self):

        out = self.subject_compute.run()
        self.assertEqual(out.shape[0], 63, 'There are not 63 rows in the outputted dataframe')

    def test_xx_weight_decisions_constant(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs(n_trials=5)
            rand_decs2 = self.compute.weight_decisions(rand_decs, decay='constant')
            self.assertListEqual(rand_decs.tolist(), rand_decs2.tolist(), 'The constant weighting is off: raw decisions should equal constant weighted (weight=1) decisions')

    def test_xx_cumulative_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs(n_trials=5)
            rand_decs1, rand_coords = self.compute.cumulative_coords(rand_decs)
            self.assertListEqual(rand_decs.tolist(), rand_decs1.tolist(), 'cumulative_coords changed the decisions')
            self.assertEqual(np.sum(rand_decs[:, 0]), rand_coords[-1, 0], 'The last affil coordinate is not what it should be')
            self.assertEqual(np.sum(rand_decs[:, 1]), rand_coords[-1, 1], 'The last power coordinate is not what it should be')

    def test_xx_previous_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs(n_trials=5)
            self.assertEqual(rand_decs.shape, (5,2))

            prev_decisions, prev_coords = self.compute.previous_coords(rand_decs, by=1)
            self.assertEqual(prev_decisions.shape, (5,2))
            self.assertEqual(prev_coords.shape, (5,2))

            # previous_decisions/coordinates row 1 should = 0, 0
            self.assertListEqual(prev_decisions[0,:].tolist(), [0,0], 'Previous decisions in row 0 do not equal [0,0]')
            self.assertListEqual(prev_coords[0,:].tolist(), [0,0], 'Previous coordinates in row 0 do not equal [0,0]')

            # previous_decisions/coordinates [1:end] = decisions/coordinates [0:end-1]
            self.assertListEqual(prev_decisions[1:].tolist(), rand_decs[0:-1].tolist(), 'Previous coordinates in row 1:end do not equal coordinates row 0:end-1')
    
    def test_xx_counterfactual_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs(n_trials=5)
            # rand_coords = np.cumsum(rand_decisions, axis=0)
            self.assertEqual(rand_decs.shape, (5,2))

            cf_decisions, cf_coords = self.compute.counterfactual_coords(rand_decs)
            self.assertEqual(cf_decisions.shape, (5,2))
            self.assertEqual(cf_coords.shape, (5,2))
            self.assertListEqual((-rand_decs).tolist(), (cf_decisions).tolist(), 'Countefactual decisions are wrongly computed')

    def test_xx_cumulative_mean(self):

        rand_decs = self.get_random_decs(n_trials=5)
        resp_mask = rand_decs != 0
        cum_mean, (cum_sum, cum_count) = self.compute.cumulative_mean(rand_decs, resp_mask)
        self.assertEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], 'Cumulative mean for 1st dimension (ie, affiliation) is off')
        self.assertEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], 'Cumulative mean for 2nd dimension (ie, power) is off')

    def test_xx_minmax_coords(self):
        rand_decs = self.get_random_decs(n_trials=5) # 5,2
        rand_coords = np.cumsum(rand_decs, axis=0)
        inconsist, consist = self.compute.minmax_coords(rand_decs) # between 0 & 1
        self.assertEqual(np.sum(consist >= inconsist), 10, 'Some of the inconsistent pattern coordinates (min) are not equal or smaller to consistent pattern')
    # def test_xx_cumulative_consistency(self):
    
    def test_xx_add_3rd_dimension_output_shape(self):
        for _ in range(10):
            U = np.random.randint(1,10, size=(np.random.randint(1,10),2)) # all the outputs should have this shape
            V = np.random.randint(1,10, size=(2,))
            ori = np.array([[0,0]])   
            U, V, ori = self.compute.add_3rd_dimension(U, V, ori)
            self.assertEqual(U.shape, V.shape, f'Adding in 3rd dimension made mismatched vector shapes {U.shape} != {V.shape}')  

    def test_xx_quadrant_overlap_sum1_and_correct_quad(self):

        q1_coords = np.array([[4,4], [-1,4], [-1,-1], [4,-1]])
        q2_coords = q1_coords - np.repeat([[3,0]], 4, axis=0) # x-3
        q3_coords = q2_coords - np.repeat([[0,3]], 4, axis=0) # y-3
        q4_coords = q3_coords + np.repeat([[3,0]], 4, axis=0) # x+3

        for q, qC in enumerate([q1_coords, q2_coords, q3_coords, q4_coords]):

            overlap = self.compute.quadrant_overlap(qC) # SLOW - speed up

            summed = np.sum(overlap)
            self.assertEqual(summed, 1, f'The overlap sum {summed}!=1')

            max_quad = np.where(overlap == np.max(overlap))[0][0]
            self.assertEqual(max_quad, q, f'The quarant with the max value {max_quad}!={q}')
        

    # ori & U should have z-axis coordinates that increase sequentially; V should have a fixed z-axis coordinate

if __name__ == '__main__':
    unittest.main()