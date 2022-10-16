import os, sys, glob, warnings, math, patsy, csv
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import zscore
from scipy.spatial import ConvexHull
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from PIL import Image
from pathlib import Path

pkg_dir = str(Path(__file__).parent.absolute())
from info import *
from utils import *

from toolbox.circ_stats import * 
from toolbox.math import * 
from toolbox.matrices import * 
from toolbox.utils import *


##########################################################################################
# parse snt logs, jsons 
##########################################################################################

def parse_log(file_path, experimenter, output_timing=True, out_dir=None): 
    '''
        Parse social navigation cogent logs & generate excel sheets

        Arguments
        ---------
        file_path : str
            Path to the log file
        experimenter : _type_
            Button numbers changed depending on the experiment
        output_timing : bool (optional, default=True)
            Set to true if want timing files 
        out_dir : str (optional, default=None)
            Specify the output directory 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # directories
    if out_dir is None: out_dir = Path(f'{os.getcwd()}/preprocessed_behavior')
    xlsx_dir = Path(f'{out_dir}/organized')
    for dir_ in [out_dir, xlsx_dir]:
        if not os.path.exists(dir_): 
            os.makedirs(dir_)
    timing_dir = Path(f'{out_dir}/timing/')
    if output_timing & (not os.path.exists(timing_dir)):
        os.makedirs(timing_dir)

    file_path = Path(file_path)
    sub_id = re.split('_|\.', file_path.name)[1] # expects a file w/ snt_subid

    # key presses differed across iterations of the task: 
    # - these versions had a fixed choice order across subjects
    if experimenter == 'RT': 
        keys = ['30','31'] # 1,2
        tr_key = '63'
    elif experimenter in ['AF','RR']: 
        keys = ['28','29']
    elif experimenter in ['NR','CS','KB','FF']: 
        keys = ['29','30']
        tr_key = '54'

    # Read input data into data variable - a list of all the rows in input file
    # Each data row has 4 or 8 items, for example:
        #['432843', '[1]', ':', 'Key', '54', 'DOWN', 'at', '418280  ']
        #['384919', '[3986]', ':', 'slide_28_end: 384919']
    with open(file_path, 'r') as csvfile: 
        data = [row for row in csv.reader(csvfile, delimiter = '\t') if len(row) == 8 or len(row) == 4]

    # parse log into a standardized xlsx
    choice_data = pd.DataFrame()

    # find the first slide onset --> eg, ['50821', '[11]', ':', 'pic_1_start: 50811']
    first_img = [row for row in data if row[3].startswith('pic_1_start')][0] # the first time the first character's image is shown
    task_start = int(first_img[3].split()[1])

    for t,trial in decision_trials.iterrows():

        row_num = -1
        found   = False # If valid button push has been found on the relevant slide

        # find the row numbdr for this trial 
        while row_num < len(data):
            row_num += 1
            if data[row_num][3].startswith("%s_start" % trial['cogent_slide_num']):
                break

        # slide start onset
        if row_num < len(data):
            slide_start = int(data[row_num][3].split()[1])
            slide_onset = (slide_start - int(task_start))/1000
            slide_end   = int(slide_start) + 11988 # slide end is 11988ms after start
        else:
            print('ERROR: %s_start not found!' % trial['cogent_slide_num'])

        # find valid button push ('Key DOWN' slide with a normal RT)
        while row_num < len(data):
            row_num += 1

            if (data[row_num][3] == 'Key' and data[row_num][5] == 'DOWN') and \
            (data[row_num][4] == keys[0] or data[row_num][4] == keys[1]):

                # want to make sure rt is within response window
                if int(data[row_num][7]) > int(slide_start) and int(data[row_num][7]) < slide_end: 
                    found = True
                    key = data[row_num][4]
                    rt  = int(data[row_num][7]) - int(slide_start)
                    bp  = (1 if key == keys[0] else 2) 
                    dec = int(trial['cogent_opt1'] if key == keys[0] else trial['cogent_opt2'])

            # If slide ends before finding valid button push
            elif "_end" in data[row_num][3]: 
                if not found:
                    rt  = 0
                    bp  = 0
                    dec = 0
                break

        # output
        if trial['dimension'] == 'affil': dim_decs = [dec, 0]
        else:                             dim_decs = [0, dec]
        choice_data.loc[t, ['decision_num','onset','button_press','decision','affil','power','reaction_time']] = [t+1, slide_onset, bp, dec] + dim_decs + [rt/1000]

    convert_dict = {'decision_num': int,
                    'onset': float,
                    'dimension': str,
                    'scene_num': int,
                    'char_role_num': int,
                    'char_decision_num': int,
                    'button_press': int,
                    'decision': int,
                    'affil': int,
                    'power': int,
                    'reaction_time': float}
    choice_data = decision_trials[['decision_num','dimension','scene_num','char_role_num','char_decision_num']].merge(choice_data, on='decision_num')
    choice_data = choice_data.astype(convert_dict)

    choice_data.to_excel(Path(f'{xlsx_dir}/snt_{sub_id}.xlsx'), index=False)


    if output_timing:

        onsets = []
        offsets = []
        trs = []

        for row in data: 
            if all(r in row for r in ['Key', tr_key, 'DOWN']):
                trs.append(int(row[0].split(': ')[0]))
            start = [r for r in row if 'start' in r]
            if len(start) > 0:
                onsets.append(start[0].split(': '))
            end = [r for r in row if 'end' in r]
            if len(end) > 0:
                offsets.append(end[0].split(': ')) 

        # will be 1 more offset than on, so do separately and then merge on the slide number
        onsets = np.array(onsets)
        onsets[:,0] = [txt.split('_start')[0] for txt in onsets[:,0]]            
        onsets_df = pd.DataFrame(onsets, columns = ['slide', 'onset_raw'])

        offsets = np.array(offsets)
        offsets[:,0] = [txt.split('_end')[0] for txt in offsets[:,0]]
        offsets_df = pd.DataFrame(offsets, columns = ['slide', 'offset_raw'])

        timing_df = onsets_df.merge(offsets_df, on='slide')
        timing_df[['onset_raw', 'offset_raw']] = timing_df[['onset_raw', 'offset_raw']].astype(int)
        timing_df.sort_values(by='onset_raw', inplace=True) 

        time0 = int(onsets_df['onset_raw'][0])
        timing_df[['onset', 'offset']] = (timing_df[['onset_raw', 'offset_raw']] - time0) / 1000 # turn into seconds
        timing_df['duration'] = timing_df['offset'] - timing_df['onset']
        timing_df = timing_df[(timing_df['duration'] < 13) & (timing_df['duration'] > 0)] # removes annoying pic slide duplicates...
        timing_df.reset_index(drop=True, inplace=True)

        timing_df.insert(1, 'trial_type', task['trial_type'].values.reshape(-1,1))

        assert timing_df['onset'][0] == 0.0, f'WARNING: {sub_id} first onset is off'
        assert timing_df['offset'].values[-1] < 1600, f'WARNING: {sub_id} timing seems too long'
        assert np.sum(timing_df['duration'] > 11) == 63, f'WARNING: {sub_id} number of decisions are not 63'

        timing_df.to_excel(Path(f'{timing_dir}/snt_{sub_id}_timing.xlsx'), index=False)

##########################################################################################
# parse snt dots jpgs
##########################################################################################

def process_dots(img):
    img = load_image(img)
    recon_img, coords_df = define_char_coords(img)
    return recon_img, coords_df


def load_image(img): 
    return Image.open(img)


def get_dot_coords(img, plot=False):
    
    with warnings.catch_warnings():
        
        # binarize image 
        binary_img = (img[:,:,1] > 0) * 1 # 3d -> 2d 
        erod_img   = ndimage.binary_erosion(binary_img, iterations=3) # erode to get rid of specks
        recon_img  = ndimage.binary_propagation(erod_img, mask=erod_img) * 1 # fill in 

        # segment image
        # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_spectral_clustering.html#sphx-glr-advanced-image-processing-auto-examples-plot-spectral-clustering-py
        # Convert the image into a graph with the value of the gradient on the edges
        graph = image.img_to_graph(binary_img, mask=recon_img.astype(bool))

        # Take a decreasing function of the gradient: we take it weakly
        # dependant from the gradient the segmentation is close to a voronoi
        graph.data = np.exp(-graph.data / graph.data.std())

        try: 
             # Force the solver to be arpack, since amg is numerically unstable
            labels   = spectral_clustering(graph, n_clusters=4)
            label_im = -np.ones(binary_img.shape)
            label_im[recon_img.astype(bool)] = labels

            # re-binarize image
            dot_im = label_im > 0
            ys, xs = np.where(dot_im == 1) # reversed
            x, y = xs[int(round(len(xs)/2))], ys[int(round(len(ys)/2))]

        except: 
            
            x, y = np.nan, np.nan
    
    if plot:

        plt.imshow(dot_im, cmap=plt.cm.nipy_spectral, interpolation='nearest')
        plt.show()
    
    return x, y


def define_char_coords(img):

    # note: the powerpoint was hardcoded with the character name, not the role (which varied across versions)
    width, height = img.size
    rgb_img = img.convert('RGB') 

    # character colors
    character_colors = {
        'peter'   : (255, 159, 63), #orange
        'olivia'  : (31, 159, 95), #green
        'newcomb' : (255, 255, 31), #yellow
        'hayworth': (159, 159, 159), #grey
        'kayce'   : (191, 159, 127), #brown
        'anthony' : (63, 127, 191), #blue
        # 'pov'     : (236, 49, 56) #red
    }

    # each character gets own img
    character_maps = {
        'peter'   : np.full((height, width, 3), 0, dtype = np.uint8),
        'olivia'  : np.full((height, width, 3), 0, dtype = np.uint8),
        'newcomb' : np.full((height, width, 3), 0, dtype = np.uint8),
        'hayworth': np.full((height, width, 3), 0, dtype = np.uint8),
        'kayce'   : np.full((height, width, 3), 0, dtype = np.uint8),
        'anthony' : np.full((height, width, 3), 0, dtype = np.uint8),        
        # 'pov'     : np.full((height, width, 3), 0, dtype = np.uint8),
    }

    # iterate over all pixels
    for w in range(width):
        for h in range(height):
            current_rgb = rgb_img.getpixel((w, h))
            curr_r, curr_g, curr_b = current_rgb
            for name, rgb in character_colors.items():
                r, g, b = rgb
                adj = 40 # allow for a little color range
                if ((r - adj) <= curr_r <= (r + adj)) and ((g - adj) <= curr_g <= (g + adj)) and ((b - adj) <= curr_b <= (b + adj)):
                    character_maps[name][h, w] = rgb 

    # get coordinates
    coords = np.array([get_dot_coords(img_) for _, img_ in character_maps.items()]).astype(float)

    # scale coordinates between -1 & 1
    coords_norm = np.zeros_like(coords)
    coords_norm[:,0] = (coords[:,0] - (w * .1) - (.5 * h))/ (.5 * h) # adjust to get rid of text space, then scale
    coords_norm[:,1] = (.5 * h - coords[:,1])/ (.5 * h)
    coords_norm = coords_norm.reshape(1,-1)

    # reconstructed image
    recon_img = (character_maps['olivia'] + character_maps['peter'] + character_maps['newcomb'] + character_maps['hayworth'] + character_maps['anthony'] + character_maps['kayce'])
    recon_img = np.where(recon_img==[0,0,0], [255,255,255], recon_img).astype(np.uint8)

    # dataframe
    headers = ['Peter_affil', 'Peter_power', 'Olivia_affil', 'Olivia_power', 'Newcomb_affil', 'Newcomb_power', 
               'Hayworth_affil', 'Hayworth_power', 'Kayce_affil', 'Kayce_power','Anthony_affil', 'Anthony_power']
    coords_df = pd.DataFrame(coords_norm, columns=headers)

    return recon_img, coords_df

##########################################################################################
# compute behavioral variables
##########################################################################################

def _shift_down(arr, by=1, replace=0):
    padded = np.ones_like(arr) * replace
    padded[by:] = np.array(arr)[0:-by] # replace 
    return padded


def minmax_coords(decisions):
    ''' 
        find the max & min coordinates 
        incrementally accounts for non-responses
        incremental mean: accumulated choices / number of choices made, at each time point
    '''
    decisions = np.array(decisions)
    
    # most consistency possible [for decision pattern]
    con = abs(decisions)
    con_coords = np.cumsum(con, 0)

    # least consistency possible [for decision pattern]
    incon = np.zeros_like(con)
    for ndim in range(2):
        mask = con[:, ndim] != 0 
        con_d = con[mask, ndim] # if its not 0, its a response
        incon[mask, ndim] = [n if not i % 2 else -n for i,n in enumerate(con_d)] # flip signs
    incon_coords = np.cumsum(incon, 0)

    # want the cumulative mean to control for non-responses: divide by num of responses to that point
    resp_counts = np.cumsum(abs(decisions) != 0, 0) 

    return [incon_coords / resp_counts, con_coords / resp_counts]


def compute_behavior(file_path, out_dir=None):
    
    # annoying warnings that I dont think matter really
    from warnings import simplefilter
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # fragmented df
    np.seterr(divide='ignore', invalid='ignore') # division by 0 in some of our operations

    # out directory
    if out_dir is None: 
        out_dir = Path(f'{os.getcwd()}/preprocessed_behavior/behavior')        
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
            
    ### load in data ###
    sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
    assert is_numeric(sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

    file_path = Path(file_path)
    if file_path.suffix == '.xlsx':  behavior = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.suffix == '.xls': behavior = pd.read_excel(file_path)
    elif file_path.suffix == '.csv': behavior = pd.read_csv(file_path)


    #####################################
    ### compute things w/in character ###
    #####################################

    for c in [1,2,3,4,5,9]:

        mask_df = (behavior['char_role_num']==c).values
        ixs = np.where(behavior['char_role_num']==c)[0]
        behav = behavior.loc[ixs,:]
        decisions = behav[['affil','power']].values

        # dont count non-responses for averages, etc
        responded  = (behav['button_press'] != 0).values.reshape(-1,1)
        mask_affil = (behav['dimension'] == 'affil').values
        mask_power = (behav['dimension'] == 'power').values
        resp_mask  =  np.vstack([mask_affil, mask_power]).T * responded

        assert decisions.shape == resp_mask.shape, 'decision and response mask arrays are diff shapes'
        tests = {'Df mask': [mask_df, (63,)]}
        for testname, test in tests.items():
             assert test[0].shape == test[1], f'{testname} should have shape {test[1]} but has shape ({test[0].shape})'


        ################################
        ###  decisions & coordinates ###
        ################################

        # different decision weighting schemes
        nt = len(decisions) 
        constant    = np.ones(nt)[:,None] # standard weighting with a constant
        linear      = linear_decay(1, 1/nt, nt)[:,None]
        exponential = exponential_decay(1, 1/nt, nt)[:,None]
        
        for wt, w in {'':constant, '_linear-decay':linear, '_expon-decay':exponential}.items():

            ### weight the decisions & make coordinates 
            decisions_w = decisions * w
            coords_w = np.cumsum(decisions_w, 0)

            ### current decisions & coordinates
            # - eg, if on each trial the subj represents the the decision/coordinates they chose
            behavior.loc[ixs, [f'affil{wt}', f'power{wt}']] = decisions_w.astype(float)
            behavior.loc[ixs, [f'affil_coord{wt}', f'power_coord{wt}']] = coords_w.astype(float)

            ### previous decisions & coordinates
            # - eg, if on each trial the subj represents the last chosen decision/coordinates
            shifted_decisions_w = _shift_down(decisions_w, by=1) 
            behavior.loc[ixs, [f'affil{wt}_prev', f'power{wt}_prev']] = shifted_decisions_w
            behavior.loc[ixs, [f'affil_coord{wt}_prev', f'power_coord{wt}_prev']] = np.cumsum(shifted_decisions_w, 0)

            ### countefactual decisions & coordinates
            # - counterfactual wrt each trial
            # -- eg, if on each trial the subj represents the [ultimately] non-chosen decision/coordinates
            prev_coords_w = coords_w - decisions_w # for each trial, undo the decisions by subtracting to get the prev coords
            decisions_w_cf = -decisions_w # opposite decision
            behavior.loc[ixs, [f'affil{wt}_cf',f'power{wt}_cf']] = decisions_w_cf
            behavior.loc[ixs, [f'affil_coord{wt}_cf',f'power_coord{wt}_cf']] = prev_coords_w + decisions_w_cf # flip the decision made, and add back in 


            ###########################
            ### geometric variables ###
            ###########################

            # loop over each decision type
            for dt in ['','_prev','_cf']:

                decisions = behavior.loc[ixs, [f'affil{wt}{dt}',f'power{wt}{dt}']].values
                coords    = behavior.loc[ixs, [f'affil_coord{wt}{dt}',f'power_coord{wt}{dt}']].values

                ### cumulative mean along each dimension [-1,+1] ###
                cum_mean = coords / np.cumsum(resp_mask, 0) # divide each time point by the response count 
                if c != 9: # neu will get all nans
                    assert np.all(cum_mean[-1,:] == coords[-1,:] / 6), 'cumulative mean is off'

                behavior.loc[ixs, [f'affil_mean{wt}{dt}', f'power_mean{wt}{dt}']] = cum_mean

                ### culumative consistency: how far did they go in the direction they were traveling? [0,1] ###

                minmax = minmax_coords(decisions)

                # 1d consistency = abs value coordinate, scaled by min and max possible coordinate            
                behavior.loc[ixs, [f'affil_consistency{wt}{dt}', f'power_consistency{wt}{dt}']] = (np.abs(cum_mean) - minmax[0]) / (minmax[1] - minmax[0])

                # 2d consistency = decision vector length, scaled by min and max possible vector lengths
                min_r = np.array([l2_norm(v) for v in minmax[0]])
                max_r = np.array([l2_norm(v) for v in minmax[1]])
                r = np.array([l2_norm(v) for v in cum_mean])
                behavior.loc[ixs, f'consistency{wt}{dt}'] = (r - min_r) / (max_r - min_r)

                ### multi-dimensional: angles & distances ###

                # directional angles between (ori to poi) and (ori to ref) [optional 3rd dimension]
                # - origin (ori)
                # --- neu: (0, 0, [interaction # (1:12)]) - note that 'origin' moves w/ interactions if in 3d
                # --- pov: (6, 0, [interaction # (1:12)])
                # - reference vector (ref)
                # --- neu: (6, 0, [max interaction (12)])
                # --- pov: (6, 6, [max interaction (12)])
                # - point of interaction vector (poi): (curr. affil coord, power coord, [interaction # (1:12)])
                # to get directional vetctors (poi-ori), (ref-ori)

                ref_frames = {'neu': {'ori': np.array([[0,0]]), 'ref': np.array([[6,0]]), 'dir': False},
                              'pov': {'ori': np.array([[6,0]]), 'ref': np.array([[6,6]]), 'dir': None}} 
                int_num    = np.arange(1, len(behav)+1).reshape(-1, 1)

                for ndim in np.arange(2, 4):    
                    for ot in ['neu', 'pov']:

                        ## angles between ori-poi & ori-ref
                        ref_frame = ref_frames[ot]
                        poi = coords
                        ref = ref_frame['ref']
                        ori = ref_frame['ori']
                        drn = ref_frame['dir'] # direction for angle calc

                        # if 3d, add 3rd dimension
                        if ndim == 3: 
                            poi = np.concatenate([poi, int_num], 1) # changes w/ num of interactions
                            ref = np.repeat(np.hstack([ref[0], 12]).reshape(1, -1), len(poi), 0) # fixed
                            ori = np.concatenate([np.repeat(ori.reshape(1, -1), len(poi), 0), int_num], 1) # changes w/ num of interactions
                            drn = None # may not be correct for neutral origin... not sure yet

                        # account for origin
                        poi = poi - ori
                        ref = ref - ori

                        # angle between ori-poi & ori-ref
                        behavior.loc[ixs, f'{ot}{ndim}d_angle{wt}{dt}'] = calculate_angle(poi, ref, direction=drn) # outputs radians

                        ## distances from ori-poi
                        if ndim == 2: 
                            behavior.loc[ixs, f'{ot}_distance{wt}{dt}'] = [l2_norm(v) for v in poi] # l2 norm is euclidean distance from ori

    ######################################
    ## variables across all characters ###
    ######################################

    suffixes = flatten_nested_lists([[f'{wt}{dt}' for dt in ['','_prev','_cf'] for wt in ['', '_linear-decay', '_expon-decay']]])
    
    for sx in suffixes:

        quadrants = {'1': np.array([[0,0], [0,6],  [6,0],  [6,6]]),   '2': np.array([[0,0], [0,6],  [-6,0], [-6,6]]),
                     '3': np.array([[0,0], [0,-6], [-6,0], [-6,-6]]), '4': np.array([[0,0], [0,-6], [6,0],  [6,-6]])}

        for t in range(0, 63):

            # in 3d
            X = behavior.loc[:t, [f'affil_coord{sx}', f'power_coord{sx}', 'char_decision_num']].values

            try:    vol = ConvexHull(X).volume
            except: vol = np.nan

            # in 2d
            try:    

                shape = ConvexHull(X[:,0:2])
                perim = shape.area # perimeter
                area  = shape.volume  # area
                poly  = Polygon(X[:,0:2][shape.vertices]) 
                overlap = []
                for q, C in quadrants.items(): # overlap w/ quadrants
                    q = Polygon(C[ConvexHull(C).vertices]) 
                    overlap.append(poly.intersection(q).area/poly.area)
            except: 

                perim = np.nan
                area  = np.nan
                overlap = [np.nan,np.nan,np.nan,np.nan]

            behavior.loc[t, f'perimeter{sx}']  = perim
            behavior.loc[t, f'area{sx}']       = area
            behavior.loc[t, f'volume{sx}']     = vol
            behavior.loc[t, f'Q1_overlap{sx}'] = overlap[0]
            behavior.loc[t, f'Q2_overlap{sx}'] = overlap[1]
            behavior.loc[t, f'Q3_overlap{sx}'] = overlap[2]
            behavior.loc[t, f'Q4_overlap{sx}'] = overlap[3]
            
            
    behavior.to_excel(Path(f'{out_dir}/snt_{sub_id}_behavior.xlsx'), index=False)

          
def summarize_behavior(file_paths, out_dir=None):
    
    if out_dir is None: 
        out_dir = Path(f'{os.getcwd()}/preprocessed_behavior')
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    summaries = []

    file_paths = sorted((f for f in file_paths if (not f.startswith(".")) & ("~$" not in f)), key=str.lower) # ignore hidden files & sort alphabetically
    for s, file_path in enumerate(file_paths):
        print(f'Preprocessing {s+1} of {len(file_paths)}', end='\r')

        ### load in data ###
        file_path = Path(file_path)
        if file_path.suffix == '.xlsx':  behavior = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.suffix == '.xls': behavior = pd.read_excel(file_path)
        elif file_path.suffix == '.csv': behavior = pd.read_csv(file_path)

        sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
        assert is_numeric(sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

        summary = pd.DataFrame()
        summary.loc[0, 'sub_id'] = sub_id
        summary.loc[0, 'reaction_time_mean'] = np.mean(behavior['reaction_time'])
        summary.loc[0, 'reaction_time_std']  = np.std(behavior['reaction_time'])
        summary.loc[0, 'missed_trials'] = np.sum(behavior['button_press'] == 0) # only missed trials for in-person
        summary.loc[0, [f'decision_{d:02d}' for d in range(1,64)]] = behavior['decision'].values


        excl_cols = ['decision_num', 'dimension', 'scene_num', 'char_role_num', 
                     'char_decision_num', 'onset', 'button_press', 'decision', 'reaction_time', 'affil',
                     'power','affil_coord','power_coord','affil_prev','power_prev',
                     'affil_coord_prev','power_coord_prev','affil_cf','power_cf','affil_coord_cf','power_coord_cf']
        cols = [c for c in behavior.columns if c not in excl_cols]

        # means of all trials
        for c,col in enumerate(cols):
            if 'angle' not in col: summary.loc[0, col + '_mean'] = np.mean(behavior[col])
            else:                  summary.loc[0, col + '_mean'] = circ_mean(behavior[col])

        # last trial only
        end_df = pd.DataFrame(behavior.loc[62,cols].values).T
        end_df.columns = [c + '_end' for c in cols]

        summary = pd.concat([summary, end_df], axis=1)

        summaries.append(summary)

    summary = pd.concat(summaries)
    summary.to_excel(Path(f'{out_dir}/SNT_summary_n{summary.shape[0]}.xlsx'), index=False)

##########################################################################################
# compute mvpa 
##########################################################################################

def get_rdv_trials(trial_ixs, rdm_size=63):

    # fill up a dummy rdm with the rdm ixs
    rdm  = np.zeros((rdm_size, rdm_size))
    rdm_ixs = combos(trial_ixs, k=2)
    for i in rdm_ixs: 
        rdm[i[0],i[1]] = 1
        rdm[i[1],i[0]] = 1
    rdv = symm_mat_to_ut_vec(rdm)
    
    return (rdv == 1), np.where(rdv==1)[0] # boolean mask, ixs


def get_char_rdv(char_int, trial_ixs=None, rdv_to_mask=None):
    ''' gets a categorical rdv for a given character (represented as integers from 1-5)
        should make more flexible to also be able to grab 
        NOTE: this is the upper triangle
    '''
    
    if trial_ixs is not None:
        decisions = decision_trials.loc[trial_ixs,:].copy()        
    else:
        decisions = decision_trials
    
    char_rdm = np.ones((decisions.shape[0], decisions.shape[0]))
    char_ixs = np.where(decisions['char_role_num'] == char_int)[0]
    char_rdv = get_rdv_trials(char_ixs, rdm_size=decisions.shape[0])[0] * 1
    
    # if want another rdv to be subsetted
    if rdv_to_mask is not None:
        rdv_to_mask = rdv_to_mask.copy()
        assert char_rdv.shape == rdv_to_mask.shape, f'the shapes are mismatched: {char_rdv.shape} {rdv_to_mask.shape}'
        char_rdv = rdv_to_mask[char_rdv==0].values 
        
    return char_rdv


def get_ctl_rdvs(metric='euclidean', trial_ixs=None):
    
    # covariates: same across everyone 
    # maybe just store it somehwerre and grab
    # upper triangles

    if trial_ixs is not None: 
        decisions = decision_trials.loc[trial_ixs,:]
    else:
        decisions = decision_trials
    cols = []
    
    # time-related drift rdms - continuous-ish
    time_rdvs = np.vstack([ut_vec_pw_dist(np.array(decisions['cogent_onset'])) ** p for p in range(1,8)]).T
    cols = cols + [f'time{t+1}' for t in range(time_rdvs.shape[1])]

    # narrative rdms - continuous-ish
    narr_rdvs = np.vstack([ut_vec_pw_dist(decisions[col].values) for col in ['slide_num','scene_num','char_decision_num']]).T
    cols = cols + ['slide','scene','familiarity']

    # dimension rdms - categorical 
    dim_rdv = ut_vec_pw_dist(np.array((decisions['dimension'] == 'affil') * 1).reshape(-1,1), metric=metric) # diff or same dims?
    dim_rdvs = []
    for dim in ['affil', 'power']: # isolate each dim
        dim_ixs = np.where(decisions['dimension'] == dim)[0]    
        dim_rdvs.append(get_rdv_trials(dim_ixs, rdm_size=len(decisions))[0] * 1)
        
    dim_rdvs = np.vstack([dim_rdvs, dim_rdv]).T
    cols = cols + ['affiliation','power','dimension']

    # character rdms - categorical
    char_rdvs = np.array([list(get_char_rdv(c, trial_ixs=trial_ixs)) for c in range(1,6)]).T
    cols = cols + ['char1', 'char2', 'char3', 'char4', 'char5']

    return pd.DataFrame(np.hstack([time_rdvs, narr_rdvs, dim_rdvs, char_rdvs]), columns=cols)


def compute_rdvs(file_path, metric='euclidean', output_all=True, out_dir=None):

    # out directory
    if out_dir is None: 
        out_dir = Path(f'{os.getcwd()}/preprocessed_behavior/mvpa')        
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    ### load in data ###
    sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
    assert is_numeric(sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

    file_path = Path(file_path)
    if file_path.suffix == '.xlsx':  behavior_ = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.suffix == '.xls': behavior_ = pd.read_excel(file_path)
    elif file_path.suffix == '.csv': behavior_ = pd.read_csv(file_path)

    # output all the decision type models?
    if output_all: 
        suffixes = flatten_nested_lists([[f'{wt}{dt}' for dt in ['','_prev','_cf'] for wt in ['', '_linear-decay', '_expon-decay']]]) 
    else: 
        suffixes = ''
        
    for sx in suffixes: 

        behavior     = behavior_[['decision', 'reaction_time', 'button_press', 'char_decision_num', 'char_role_num',f'affil{sx}',f'power{sx}',f'affil_coord{sx}',f'power_coord{sx}']]
        end_behavior = behavior[behavior['char_decision_num'] == 12].sort_values(by='char_role_num')

        for outname, behav in {sx: behavior, f'{sx}_end': end_behavior}.items(): 

            decisions = np.sum(behav[[f'affil{sx}',f'power{sx}']],1)
            coords    = behav[[f'affil_coord{sx}',f'power_coord{sx}']].values

            rdvs = get_ctl_rdvs(trial_ixs=behav.index)
            rdvs.loc[:,'reaction_time'] = ut_vec_pw_dist(np.nan_to_num(behav['reaction_time'], 0))
            rdvs.loc[:,'button_press']  = ut_vec_pw_dist(np.array(behav['button_press']))

            ######################################################
            # relative distances between locations
            # - can try other distances: e.g., manhattan which would be path distance
            ######################################################

            metric = 'euclidean'
            rdvs.loc[:,'place_2d']       = ut_vec_pw_dist(coords, metric=metric)
            rdvs.loc[:,'place_affil']    = ut_vec_pw_dist(coords[:,0], metric=metric)
            rdvs.loc[:,'place_power']    = ut_vec_pw_dist(coords[:,1], metric=metric)
            rdvs.loc[:,'place_positive'] = ut_vec_pw_dist(np.sum(coords, 1), metric=metric)

            #     # newer adds:
            #     rdvs['place_2d_scaled', ut_vec_pw_dist(behavior[['affil_coord_scaled', 'power_coord_scaled']])) # dont zscore cuz already scaled
            #     rdvs['place_2d_exp_decay', ut_vec_pw_dist(behavior[['affil_coord_exp-decay', 'power_coord_exp-decay']]))
            #     rdvs['place_2d_exp_decay_scaled', ut_vec_pw_dist(behavior[['affil_coord_exp-decay_scaled', 'power_coord_exp-decay_scaled']]))

            ######################################################
            # distances from ref points (poi - ref)
            # -- ori to poi vector (poi - [0,0]) 
            # -- pov to poi vector (poi - [6,0]) 
            ######################################################

            for origin, ori in {'neu':[0,0], 'pov':[6,0]}.items():

                V = coords - ori

                rdvs.loc[:,f'{metric}_distance_{origin}'] = ut_vec_pw_dist(np.array([l2_norm(v) for v in V]), metric=metric)
                rdvs.loc[:,f'angular_distance_{origin}']  = symm_mat_to_ut_vec(angular_distance(V)) 
                rdvs.loc[:,f'cosine_distance_{origin}']   = symm_mat_to_ut_vec(cosine_distance(V))

            ######################################################
            # others
            ######################################################

            # decision directon: +1 or -1
            direction_rdv = ut_vec_pw_dist(behav['decision'].values.reshape(-1,1))
            direction_rdv[direction_rdv > 1] = 1 
            rdvs.loc[:,'decision_direction'] = direction_rdv

            # output
            rdvs.to_excel(Path(f'{out_dir}/snt_{sub_id}{outname}_rdvs.xlsx'), index=False)
  