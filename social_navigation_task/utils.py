import numpy as np


def remove_neutrals(arr):
    ''' remove neutral charactr trials from array (trials 15, 16, 36) '''
    return np.delete(arr, np.array([14, 15, 35]), axis=0)


def add_neutrals(arr, add=[0, 0]):
    ''' add values to the neutral character trial positions (trials 15, 16, 36) '''
    neu_arr = arr.copy()
    for row in [14, 15, 35]: # ascending order, to ensure no problems w/ shifting the array
        neu_arr = np.insert(neu_arr, row, add, axis=0)
    return neu_arr