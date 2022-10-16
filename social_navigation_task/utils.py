import numpy as np
import sklearn as sk
import itertools

#######################################################################################################
# checking 
########################################################################################################

def all_equal(iterable):
    ''' check if all items in a list are identical '''
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def all_equal_arrays(L):
    ''' check if difference in List L along axis of stacking == 0  '''
    return (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()


def is_numeric(input_str):
    ''' check if all characters in a string are numeric '''
    return all(char.isdigit() for char in input_str)


def is_finite(array):
    return array[np.isfinite(array)]


def get_unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#######################################################################################################
# neutral character
#######################################################################################################

def remove_neutrals(arr):
    ''' remove neutral charactr trials from array (trials 15, 16, 36) '''
    return np.delete(arr, np.array([14, 15, 35]), axis=0)


def add_neutrals(arr, add=[0, 0]):
    ''' add values to the neutral character trial positions (trials 15, 16, 36) '''
    neu_arr = arr.copy()
    for row in [14, 15, 35]: # ascending order, to ensure no problems w/ shifting the array
        neu_arr = np.insert(neu_arr, row, add, axis=0)
    return neu_arr

#######################################################################################################
# math
#######################################################################################################

def exponential_decay(init, decay, time_steps):
    # f(t) = i(1 - r) ** t
    return np.array([init * (1 - decay) ** step for step in range(time_steps)])


def linear_decay(init, decay, time_steps):
    # f(t) = i - r * t
    # dont let it decay past 0
    return np.array([max((0, init - decay * step)) for step in range(time_steps)])


def _coincident_vectors(u, v):
    ''' Checks if vectors (u & v) are the same or scalar multiples of each other'''
    return np.dot(u, v) * np.dot(u, v) == np.dot(u, u) * np.dot(v, v)


def angle_between_vectors(u, v, direction=None, verbose=False):
    '''
        Compute elementwise angle between sets of vectors u & v
            
        uses np.arctan2(y,x) which computes counterclockwise angle [-π, π] between origin (0,0) and x, y
        clockwise v. counterclockwise: https://itecnote.com/tecnote/python-calculate-angle-clockwise-between-two-points/  
        included: ADD LINK

        TODO: make pairwise..?

        Arguments
        ---------
        u : array-like
            vector
        v : array-like
            another vector
        direction : None, True or False (optional, default=None)
            None == Included
            True == Clockwise 360
            False == Counterclockwise 360 
        verbose : bool (optional, default=False)
             
        Returns
        -------
        float 
            angle in radians 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    #     if U.shape != V.shape:
    #         if verbose: print(f'Different shape vectors: U={U.shape}, V={V.shape}. Assuming smaller is reference.')
    #         if len(U) < len(V): U = np.repeat(np.expand_dims(U, 0), len(V), axis=0)
    #         else:               V = np.repeat(np.expand_dims(V, 0), len(U), axis=0)
    #     rads = []
    #     for u, v in zip(U, V):            
    # if one of vectors is at origin, the angle is undefined but could be considered as orthogonal (90 degrees)
    if ((u==0).all()) or ((v==0).all()): 
        if verbose: print(u, v, 'at least 1 vector at origin; treating as orthogonal')
        rad = np.pi/2

    # if same vectors (or scalar multiples of each other) being compared, no angle between (0 degrees)
    # -- b/c 0-360 degrees, direction matters: make sure the signs are the same too
    elif (_coincident_vectors(u, v)) & all(np.sign(u) == np.sign(v)):
        if verbose: print(u, v, 'same vectors, no angle in between')
        rad = 0 # 0 degrees == 360 degrees == 2*pi radians 

    else:

        if direction is None: 

            # "included" angle from [0,180], [0, π] 
            rad = np.arccos(np.dot(u, v) / (l2_norm(u) * l2_norm(v)))
            # equivalent: np.arctan2(l2_norm(np.cross(u, v)), np.dot(u, v))

        elif direction is True: 

            # clockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*v[::-1]) - np.arctan2(*u[::-1])) % (2 * np.pi)  
        
        elif direction is False:

            # counterclockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*u[::-1]) - np.arctan2(*v[::-1])) % (2 * np.pi)
            
    return rad


def calculate_angle(U, V=None, direction=None, force_pairwise=False, verbose=False):
    '''
        Calculate angles between n-dim vectors 
        If V == None, calculate U pairwise
        Else, calculate elementwise
        
        TODO: more explanation; find more elegant ways to do this; also adapt other pairwise like functions to have structure

        Arguments
        ---------
        U : array-like
            shape (n_vectors, n_dims)
        V : array-like
            shape (n_vectors, n_dims)
        direction : optional (default=None)
            None : included 180
            False : counterclockwise 360 (wont give a symmetrical matrix)
            True : clockwise 360
        force_pairwise : optional (default=False)
            
        Returns
        -------
        numeric 
            pairwise angles in radians

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    
    # # testing (10-12-22)
    # U = np.random.randint(100, size=(4,2))
    # for V in [None, np.random.randint(100, size=U.shape), 
    # np.random.randint(100, size=(1, U.shape[1])), np.random.randint(100, size=(7, U.shape[1]))]:

    messages = []

    # check/fix shapes
    if U.ndim == 1: 
        U = np.expand_dims(U, 0)
        messages.append('Added a dimension to U')
    if V is not None:
        if V.ndim == 1: 
            V = np.expand_dims(V, 0)
            messages.append('Added a dimension to V')

    # determine output shape     
    if V is None:
        ret = 'pairwise'
        V = U 
    elif U.shape == V.shape:
        ret = 'elementwise' 
    elif (U.shape[0] > 1) & (V.shape[0] == 1): 
        V = np.repeat(V, len(U), 0) 
        ret = 'single reference'  
    messages.append(f'Calculated {ret}')
    
    # calculate angles
    radians = np.zeros((U.shape[0], V.shape[0]))
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            radians[i, j] = angle_between_vectors(U[i,:], V[j,:], direction=direction)

    # output
    if ret == 'pairwise': cols = 'U'
    else:                 cols = 'V'
    radians = pd.DataFrame(radians, index=[f'U{i+1:02d}' for i in range(len(U))], columns=[f'{cols}{i+1:02d}' for i in range(len(V))])

    if not force_pairwise: 
        if ret == 'single reference':
            radians = radians.iloc[:,0].values
        elif ret == 'elementwise':
            radians = np.diag(radians)
    if verbose: [print(m) for m in messages]
    return radians


def cosine_distance(u, v=None):
    ''' 
        cosine distance of (u, v) = 1 - (dot(u,v) / dot(l2_norm(u), l2_norm(v)))
        returns similarity measure [0,2]
    '''
    return sk.metrics.pairwise_distances(u, v, metric='cosine')


def cosine_similarity(u, v=None):
    ''' 
        cosine similarity of (u, v) = dot(u,v) / dot(l2_norm(u), l2_norm(v))
        returns similarity measure [-1,1], equivalent to [0 degrees,18 0degrees]
        maybe issue: small angles tend to get very similar values(https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance)

    '''
    return 1 - sk.metrics.pairwise_distances(u, v, metric='cosine')


def angular_distance(u, v=None):
    ''' 
        angular distance of (u, v) = arccos(cosine_similarity(u, v)) / pi
        returns dissimilarity measure [0,2]
    '''
    return np.arccos(cosine_similarity(u, v))/np.pi


def angular_similarity(u, v=None):
    ''' 
        angular similarity between two vectors = 1 - (arccos(cosine_similarity(u, v)) / pi)
        returns similarity measure [-1,1]
    '''
    return 1 - (np.arccos(cosine_similarity(u, v))/np.pi)

