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
    # TODO: add shape, type, etc testing for inputs
    # TDOD: test other methods - eg compute_character,compute_acrosscharacters 
    
    iters = 10
    n_trials = 5
    compute = preprc.ComputeBehavior(file_path=None)

    # TODO replace this! dont depend on an outside file... generate fake behavior instead?
    subject_compute = preprc.ComputeBehavior(file_path=info.example_xlsx, out_dir=False)

    def get_random_decs(self):
        return np.array([random.choice([-1,1]) for _ in range(self.n_trials*2)]).reshape(-2,2)

    def test_xx_check_input_shape(self):
        exp_shapes = [(2,2), (3,2)]
        to_check = np.array([(3,2),[3,2]])
        bool = self.compute.check_input_shape(to_check, exp_shapes)
        self.assertEqual(True, bool, 'Shape checker is off')

    # def test_xx_run_output_shape(self):

    #     out = self.subject_compute.run()
    #     self.assertEqual(out.shape[0], 63, 'There are not 63 rows in the outputted dataframe')

    def test_xx_weight_decisions_constant(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs()
            rand_decs2 = self.compute.weight_decisions(rand_decs, decay='constant')
            self.assertListEqual(rand_decs.tolist(), rand_decs2.tolist(), 'The constant weighting is off: raw decisions should equal constant weighted (weight=1) decisions')

    def test_xx_current_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs()
            rand_decs1, rand_coords = self.compute.current_coords(rand_decs)
            self.assertListEqual(rand_decs.tolist(), rand_decs1.tolist(), 'cumulative_coords changed the decisions')
            self.assertEqual(np.sum(rand_decs[:, 0]), rand_coords[-1, 0], 'The last affil coordinate is not what it should be')
            self.assertEqual(np.sum(rand_decs[:, 1]), rand_coords[-1, 1], 'The last power coordinate is not what it should be')

    def test_xx_previous_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs()
            self.assertEqual(rand_decs.shape, (self.n_trials,2))

            prev_decisions, prev_coords = self.compute.previous_coords(rand_decs, by=1)
            self.assertEqual(prev_decisions.shape, (self.n_trials,2))
            self.assertEqual(prev_coords.shape, (self.n_trials,2))

            # previous_decisions/coordinates row 1 should = 0, 0
            self.assertListEqual(prev_decisions[0,:].tolist(), [0,0], 'Previous decisions in row 0 do not equal [0,0]')
            self.assertListEqual(prev_coords[0,:].tolist(), [0,0], 'Previous coordinates in row 0 do not equal [0,0]')

            # previous_decisions/coordinates [1:end] = decisions/coordinates [0:end-1]
            self.assertListEqual(prev_decisions[1:].tolist(), rand_decs[0:-1].tolist(), 'Previous coordinates in row 1:end do not equal coordinates row 0:end-1')
    
    def test_xx_counterfactual_coords(self):

        for _ in range(self.iters):
            rand_decs = self.get_random_decs()
            # rand_coords = np.cumsum(rand_decisions, axis=0)
            self.assertEqual(rand_decs.shape, (self.n_trials,2))

            cf_decisions, cf_coords = self.compute.counterfactual_coords(rand_decs)
            self.assertEqual(cf_decisions.shape, (self.n_trials,2))
            self.assertEqual(cf_coords.shape, (self.n_trials,2))
            self.assertListEqual((-rand_decs).tolist(), (cf_decisions).tolist(), 'Countefactual decisions are wrongly computed')

    def test_xx_cumulative_mean(self):

        rand_decs = self.get_random_decs()
        resp_mask = rand_decs != 0
        cum_mean, (cum_sum, cum_count) = self.compute.cumulative_mean(rand_decs, resp_mask)
        self.assertEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], 'Cumulative mean for 1st dimension (ie, affiliation) is off')
        self.assertEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], 'Cumulative mean for 2nd dimension (ie, power) is off')

    def test_xx_simulate_consistent_behavior(self):
        rand_decs = self.get_random_decs()
        [(incon_decs, con_decs), (incon_coords, con_coords), (resp_mask, resp_counts)] = self.compute.simulate_consistent_decisions(rand_decs)

        self.assertEqual(incon_decs.shape, (self.n_trials, 2))
        self.assertEqual(con_decs.shape, (self.n_trials, 2))
        self.assertEqual(incon_coords.shape, (self.n_trials, 2))
        self.assertEqual(con_coords.shape, (self.n_trials, 2))
        self.assertEqual(resp_mask.shape, (self.n_trials, 2))
        self.assertEqual(resp_counts.shape, (self.n_trials, 2))

        self.assertEqual(np.sum(incon_coords <= con_coords), self.n_trials*2, 'Some of the inconsistent pattern coordinates (min) are not equal or smaller to consistent pattern (max)')

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