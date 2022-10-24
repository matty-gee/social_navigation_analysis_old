import os, sys, glob, warnings, re, math, patsy, csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp 
import sklearn as sk
import pycircstat
from shapely import geometry
from PIL import Image
import copy

import info 
import utils

pkg_dir = str(Path(__file__).parent.absolute())

# TODO: add little wrapper functions to let users easily run a bunch of subjects through

#------------------------------------------------------------------------------------------
# parse snt logs & csvs
#------------------------------------------------------------------------------------------


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
    if out_dir is None: out_dir = Path(os.getcwd())
    if not os.path.exists(out_dir):
        print('Creating output directory')
        os.makedirs(out_dir)

    xlsx_dir = Path(f'{out_dir}/Organized')
    if not os.path.exists(xlsx_dir):
        print('Creating subdirectory for organized data')
        os.makedirs(xlsx_dir)

    timing_dir = Path(f'{out_dir}/Timing/')
    if output_timing & (not os.path.exists(timing_dir)):
        print('Creating subdirectory for fmri timing files')
        os.makedirs(timing_dir)

    file_path = Path(file_path)
    sub_id = re.split('_|\.', file_path.name)[1] # expects a file w/ snt_subid

    # key presses differed across iterations of the task: 
    # - these versions had a fixed choice order across subjects
    experimenter = experimenter.lower()
    if experimenter == 'rt': 
        keys = ['30','31'] # 1,2
        tr_key = '63'
    elif experimenter in ['af','rr']: 
        keys = ['28','29']
    elif experimenter in ['nr','cs','kb','ff']: 
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
    info.task_start = int(first_img[3].split()[1])

    for t,trial in info.decision_trials.iterrows():

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
            slide_onset = (slide_start - int(info.task_start))/1000
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

    choice_data = merge_choice_data(choice_data)
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

        timing_df.insert(1, 'trial_type', info.task['trial_type'].values.reshape(-1,1))

        assert timing_df['onset'][0] == 0.0, f'WARNING: {sub_id} first onset is off'
        assert timing_df['offset'].values[-1] < 1600, f'WARNING: {sub_id} timing seems too long'
        assert np.sum(timing_df['duration'] > 11) == 63, f'WARNING: {sub_id} number of decisions are not 63'

        timing_df.to_excel(Path(f'{timing_dir}/snt_{sub_id}_timing.xlsx'), index=False)


def merge_choice_data(choice_data, decision_cols=None):
    if decision_cols is None:
        decision_cols = ['dimension','scene_num','char_role_num','char_decision_num']
    if 'decision_num' not in decision_cols:
        decision_cols = ['decision_num'] + decision_cols
    choice_data = info.decision_trials[decision_cols].merge(choice_data, on='decision_num')
    convert_dict = {'decision_num': int,
                    'dimension': str,
                    'scene_num': int,
                    'char_role_num': int,
                    'char_decision_num': int,
                    'button_press': int,
                    'decision': int,
                    'affil': int,
                    'power': int,
                    'reaction_time': float}
    if 'onset' in choice_data.columns:
        convert_dict['onset'] = float

    choice_data = choice_data.astype(convert_dict)
    return choice_data


class ParseCsv:
    '''
        SUMMARY
        [By Matthew G. Schafer; <mattygschafer@gmail.com>; github @matty-gee; 2020ish]
    '''
    
    def __init__(self, csv_path, snt_version='standard', verbose=0):

        self.verbose    = verbose
        self.csv        = csv_path
        self.data       = pd.read_csv(csv_path)
        self.task_ver   = self.data['task_ver'].values[0]
        
        if snt_version == 'adolescent_pilot':
            self.snt_ver = 'adolescent'
            try:
                self.sub_id = self.data.initials.values[0]
            except: 
                try: 
                    self.sub_id = self.data.prolific_id.values[0]
                except: 
                    self.sub_id = 'no_name'
        else:
            self.snt_ver = snt_version
            self.sub_id  = self.data.prolific_id.values[0]

        self.clean()

        # for older versions!
        # ordered: ['first', 'second', 'assistant', 'powerful', 'boss', 'neutral']
        self.img_sets = {'OFA': ['OlderFemaleBl_2','OlderMaleW_1','OlderMaleBr_2','OlderMaleW_4','OlderFemaleBr_3','OlderFemaleW_1'],
                            'OFB': ['OlderFemaleW_2','OlderMaleBr_1','OlderMaleW_5','OlderMaleBl_3','OlderFemaleW_3','OlderFemaleBl_1'], 
                            'OFC': ['OlderFemaleBl_2','OlderMaleBr_1','OlderMaleBr_4','OlderMaleW_5','OlderFemaleW_3','OlderFemaleW_1'], 
                            'OFD': ['OlderFemaleW_2','OlderMaleW_1','OlderMaleW_5','OlderMaleBr_3','OlderFemaleBr_3','OlderFemaleBl_1'], 
                            'OMA': ['OlderMaleBr_2','OlderFemaleW_2','OlderFemaleBr_5','OlderFemaleW_3','OlderMaleBr_1','OlderMaleW_5'], 
                            'OMB': ['OlderMaleW_1','OlderFemaleBl_2','OlderFemaleW_1','OlderFemaleBl_3','OlderMaleW_4','OlderMaleBr_4'], 
                            'OMC': ['OlderMaleBr_4','OlderFemaleBl_2','OlderFemaleBl_1','OlderFemaleW_3','OlderMaleW_3','OlderMaleW_5'], 
                            'OMD': ['OlderMaleW_1','OlderFemaleW_2','OlderFemaleW_1','OlderFemaleBr_5','OlderMaleBr_3','OlderMaleBr_4'], 
                            'YFA': ['YoungerFemaleBr_1','YoungerMaleW_4','OlderMaleBr_4','YoungerMaleW_3','OlderFemaleBr_5','OlderFemaleW_1'], 
                            'YFB': ['YoungerFemaleW_3','YoungerMaleBr_2','YoungerMaleW_2','OlderMaleBr_3','OlderFemaleW_4','OlderFemaleBl_1'], 
                            'YFC': ['YoungerFemaleBr_1','YoungerMaleBr_2','OlderMaleBr_4','OlderMaleW_4','OlderFemaleW_3','OlderFemaleW_1'], 
                            'YFD': ['YoungerFemaleW_3','YoungerMaleW_4','OlderMaleW_5','OlderMaleBr_3','OlderFemaleBr_5','OlderFemaleBl_1'],
                            'YMA': ['YoungerMaleBr_2','YoungerFemaleW_3','OlderFemaleBl_1','OlderFemaleW_4','OlderMaleBr_3','YoungerMaleW_2'],
                            'YMB': ['YoungerMaleW_4','YoungerFemaleBr_1','OlderFemaleW_1','OlderFemaleBr_5','YoungerMaleW_3','OlderMaleBr_4'],
                            'YMC': ['YoungerMaleBr_2','YoungerFemaleBr_1','OlderFemaleBl_1','OlderFemaleW_3','OlderMaleW_3','OlderMaleW_5'],
                            'YMD': ['YoungerMaleW_4','YoungerFemaleW_3','OlderFemaleW_1','OlderFemaleBr_3','OlderMaleBr_3','OlderMaleBr_4']}

    def clean(self):
        
        # data can be two identical rows for some reason
        if self.data.shape[0] > 1: 
            self.data = self.data.iloc[0,:].to_frame().T
        
        ### standardize naming conventions ###
        # there have been multiple versions of the task, multiple naming conventions etc..
        # this is an attempt to standardize the naming before extracting variables

        # make everything lower case
        self.data.columns = map(str.lower, self.data.columns)
        self.data = self.data.apply(lambda x: x.astype(str).str.lower())

        # replace character names w/ their roles
        replace_substrings = {'newcomb':'powerful', 'hayworth':'boss'}

        if 'O' in self.task_ver or 'Y' in self.task_ver:  # this doesnt apply to adolescent version...
            if 'F' in self.task_ver: 
                order = ['maya','chris','anthony','newcomb','hayworth','kayce']
            else: 
                order = ['chris','maya','kayce','newcomb','hayworth','anthony']
            for name in order: replace_substrings[name] = info.character_roles[order.index(name)]

        self.data.replace(replace_substrings, inplace=True, regex=True) # replace elements
        
        # replace column headers
        replace_substrings['.'] = '_'
        replace_substrings['narrative'] = 'snt'
        replace_substrings['demographics'] = 'judgments'
        replace_substrings['self_judgments'] = 'judgments'
        replace_substrings['relationship_feelings'] = 'character_relationship'
        for k,i in replace_substrings.items():
            self.data.columns = self.data.columns.str.replace(k,i, regex=True)
        
        # race judgments may need to be reworked
        if utils.substring_in_strings('race', self.data.columns):
            race_cols = utils.get_strings_matching_pattern(self.data.columns, 'race_*_*')
            rename = {}
            for col in race_cols:
                split_ = col.split('_')
                rename[col] = f'judgment_{split_[1]}_{split_[0]}_{split_[2]}'
            self.data.rename(columns=rename, inplace=True)
            
        return self.data
   
    def run(self):

        if 'snt_choices' not in self.data.columns:    
            print(f'{self.sub_id} does not have a "snt_choice" column. Exiting w/o preprocessing')
            return 
        else: 
            self.task_functions = {'snt': self.process_snt,
                                  'characters': self.process_characters,
                                  'memory': self.process_memory,
                                  'dots': self.process_dots, 
                                  'forced_choice': self.process_forced_choice,
                                  'ratings': self.process_ratings}     

            self.task_functions['snt']()
            post_snt = []
            for task in ['characters', 'memory', 'dots', 'ratings', 'forced_choice']:
                out = self.task_functions[task]()
                if isinstance(out, pd.DataFrame):
                    post_snt.append(out)
            self.post = pd.concat(post_snt, axis=1)
            self.post.index = [self.sub_id]
            return [self.snt, self.post]
    
    def process_snt(self):

        if 'snt_choices' not in self.data.columns:
            print(f'{self.sub_id} does not have a "snt_choice" column')
            return
        else:
            
            # the options alphabetically sorted to allow easy standardization
            validated_decisions = info.validated_decisions[self.snt_ver]

            snt_bps  = np.array([int(re.sub('["\]"]', '', d.split(':')[1])) for d in self.data['snt_choices'].values[0].split(',')]) # single column
            snt_opts = self.data['snt_opts_order'].values[0].split('","') # split on delimter
            self.snt = pd.DataFrame(columns=['decision_num', 'button_press', 'decision', 'affil', 'power'])

            for q, question in enumerate(snt_opts):

                # organize
                opt1    = utils.remove_nontext(question.split(';')[1]) # this delimeter might change?
                opt2    = utils.remove_nontext(question.split(';')[2])
                sort_ix = np.argsort((opt1, opt2)) # order options alphabetically

                # parse the choice
                choice   = sort_ix[snt_bps[q] - 1] + 1 # choice -> 1 or 2, depending on alphabetical ordering
                decision = validated_decisions.iloc[q]
                affl = np.array(decision['option{}_affil'.format(int(choice))]) # grab the correct option's affil value
                pwr  = np.array(decision['option{}_power'.format(int(choice))]) # & power
                self.snt.loc[q,:] = [q + 1, snt_bps[q], affl + pwr, affl, pwr]

            snt_rts = np.array([int(utils.remove_nonnumeric(rt)) for rt in self.data['snt_rts'].values[0].split(',')])
            self.snt['reaction_time'] = snt_rts[np.array(validated_decisions['slide_num']) - 1] / 1000
        
            self.snt = info.decision_trials[['decision_num','dimension','scene_num','char_role_num','char_decision_num']].merge(self.snt, on='decision_num')
            convert_dict = {'decision_num': int,
                            'dimension': str,
                            'scene_num': int,
                            'char_role_num': int,
                            'char_decision_num': int,
                            'button_press': int,
                            'decision': int,
                            'affil': int,
                            'power': int,
                            'reaction_time': float}
            self.snt = self.snt.astype(convert_dict) 

            # snt_df.to_excel(f'{self.data_dir}/Task/Organized/SNT_{self.sub_id}.xlsx', index=False)

            return self.snt

    def process_characters(self):
        '''
           simple classes: masculine & feminine, dark skin & light skin 
        '''
        if not utils.substring_in_strings('character_info_', self.data.columns): # older version
            img_names = [i.lower() for i in self.img_sets[self.task_ver]]
        else: # newer version
            img_names = [self.data[f'character_info_{r}_img'].values[0].lower() for r in info.character_roles]

        gender_bool    = [any([ss in i for ss in ['girl','woman','female']]) for i in img_names]
        skincolor_bool = [any([ss in i for ss in ['br','bl','brown','black','dark']]) for i in img_names]

        # make into df
        self.characters = pd.concat([pd.DataFrame(np.array(['feminine' if b  else 'masculine' for b in gender_bool])[np.newaxis], 
                                                    index=[self.sub_id], columns=[f'{r}_gender' for r in info.character_roles]),
                                     pd.DataFrame(np.array(['brown' if b  else 'white' for b in skincolor_bool])[np.newaxis], 
                                                    index=[self.sub_id], columns=[f'{r}_skincolor' for r in info.character_roles])], axis=1)
        
        return self.characters

    def process_memory(self):

        if not utils.substring_in_strings('memory', self.data.columns):
            if self.verbose: print('There are no memory columns in the csv')   
            return 
        else: 
            if self.verbose: print('Processing memory')

            # correct answers when questions are alphabetically sorted
            # {0: 'first', 1: 'second', 2: 'assistant', 3: 'newcomb', 4: 'hayworth', 5: 'neutral'}
            if self.snt_ver in ['standard', 'schema']:
                corr = [1,4,5,4,5,0,0,0,4,1,3,3,1,4,5,3,1,0,2,5,5,2,2,2,3,3,0,1,2,4] 
                # original? : [1,3,5,3,5,0,0,0,3,1,2,2,1,3,5,2,1,0,4,5,5,4,4,4,2,2,0,1,4,3]...????
            elif self.snt_ver == 'adolescent':
                corr = [1,4,0,5,5,4,0,0,0,4,3,3,3,4,1,5,3,1,2,5,2,5,1,2,2,2,3,0,1,4]
            
            memory_cols = [c for c in self.data.columns if 'memory' in c]
            if 'memory_resps' in memory_cols or 'character_memory' in memory_cols: # older version
                # these versions compressed responses into a single column with a delimeter
                try: 
                    memory_  = [t.split(';')[1:2] for t in self.data['memory_resps'].values[0].split('","')]
                except: 
                    memory_  = [t.split(';')[1:2] for t in self.data['character_memory'].values[0].split('","')]
                ques_ = [m[0].split(':')[0] for m in memory_]
                resp_ = [m[0].split(':')[1] for m in memory_]

            else: # newer version

                ques_  = self.data[[c for c in memory_cols if 'question' in c]].values[0]
                resp_  = self.data[[c for c in memory_cols if 'resp' in c]].values[0]
            
            memory = sorted(list(zip(ques_, resp_)))
            self.memory = pd.DataFrame(np.zeros((1,6)), columns=[f'memory_{cr}' for cr in info.character_roles])
            for r, resp in enumerate(memory): 
                if resp[1] == info.character_roles[corr[r]]: 
                    self.memory[f'memory_{info.character_roles[corr[r]]}'] += 1/5

            # combine summary & trial x trial
            self.memory['memory_mean'] = np.mean(self.memory.values)
            self.memory['memory_rt']   = np.mean(self.data[[c for c in memory_cols if 'rt' in c]].values[0].astype(float) / 1000)
            memory_resp_df = pd.DataFrame(np.array([r[1] for r in memory]).reshape(1, -1), 
                                          columns=[f'memory_{q + 1 :02d}_{info.character_roles[r]}' for q, r in enumerate(corr)])

            self.memory = pd.concat([self.memory, memory_resp_df], axis=1)
            self.memory.index = [self.sub_id]
            self.memory.insert(0, 'task_ver', self.data['task_ver'].values[0])
                
            return self.memory
    
    def process_dots(self):

        if not utils.substring_in_strings('dots', self.data.columns):            
            if self.verbose: print('There are no dots columns in the csv')
            return
        else: 
            if self.verbose: print('Processing dots')
            dots_cols = [c for c in self.data.columns if 'dots' in c]
            self.dots = pd.DataFrame(index=[self.sub_id], columns=[f'{c}_dots_{d}' for c in info.character_roles for d in ['affil','power']])

            # rename & standardize 
            if 'dots_resps' in dots_cols: # older version
                for row in self.data['dots_resps'].values[0].split(','):
                    split_ = row.split(';')
                    role = utils.remove_nontext(split_[0].split(':')[0])
                    self.dots[f'{role}_dots_affil'] = (float(split_[1].split(':')[1]) - 500)/500
                    self.dots[f'{role}_dots_power'] = (500 - float(split_[2].split(':')[1]))/500

            else: # newer version 
                for role in info.character_roles:
                    self.dots[f'{role}_dots_affil'] = (float(self.data[f'dots_{role}_affil'].values[0]) - 500)/500
                    self.dots[f'{role}_dots_power'] = (500 - float(self.data[f'dots_{role}_power'].values[0]))/500

            # get means
            for dim in ['affil','power']:
                self.dots[f'dots_{dim}_mean'] = np.mean(self.dots[[c for c in self.dots.columns if dim in c]],1).values[0]
                    
            return self.dots

    def process_ratings(self):
        
        if 'character_dimensions' in self.data.columns: 
            if self.verbose: print('Processing ratings (older version)')
            ratings = []
            for col in ['character_dimensions', 'character_relationship']:
                for row in [char.split(';') for char in self.data[col].values[0].split(',')]: 
                    role     = utils.remove_nontext(row[0])
                    dims     = [utils.remove_nontext(r) for r in row[1:-1]] # last is rt
                    ratings_ = [int(utils.remove_nonnumeric(r)) for r in row[1:-1]]
                    ratings.append(pd.DataFrame(np.array(ratings_)[np.newaxis], index=[self.sub_id], columns=[f'{role}_{d}' for d in dims]))
            self.ratings = pd.concat(ratings, axis=1)   
            rating_dims = np.unique([c.split('_')[1] for c in self.ratings.columns])

        elif utils.substring_in_strings('judgments', self.data.columns):
            if self.verbose: print('Processing ratings')

            rating_cols = utils.get_strings_matching_pattern(self.data.columns, 'judgments_*_resp')
            if len(self.data[rating_cols]): ratings = self.data[rating_cols].iloc[0,:].values.astype(int).reshape(1,-1)
            else:                           ratings = self.data[rating_cols].values.astype(int).reshape(1,-1)
            rating_cols  = [utils.remove_multiple_strings(c, ['judgments_','_resp']) for c in rating_cols]
            self.ratings = pd.DataFrame(ratings, index=[self.sub_id], columns=rating_cols)
            rating_dims  = np.unique([c.split('_')[1] for c in rating_cols])
            
        else:
            if self.verbose: print('There are no character/self ratings columns in the csv')
            return

        # mean of character ratings 
        for dim in rating_dims: 
            self.ratings[f'{dim}_mean'] = np.mean(self.ratings[[f'{r}_{dim}' for r in info.character_roles]], axis=1)
        
        return self.ratings

    def process_forced_choice(self):

        if not utils.substring_in_strings('forced_choice', self.data.columns): 
            if self.verbose: print('There are no forced choice columns in the csv')
            return
        else:
            if self.verbose: print('Processing forced choice')
            choices = self.data[[c for c in self.data.columns if 'forced_choice' in c]]
            n_choices = int(len(choices.columns) / 3) # 3 cols for each trial

            self.forced_choice = pd.DataFrame()
            for t in np.arange(0, n_choices):

                options = choices[f'forced_choice_{t}_comparison'].values[0].split('_&_')
                rt      = float(choices[f'forced_choice_{t}_rt'].values[0])

                # organize the responses
                resp    = float(choices[f'forced_choice_{t}_resp'].values[0]) - 50 # center
                if resp < 0: choice = options[0]
                else:        choice = options[1]

                ans                   = np.array([0, 0])
                options               = sorted(options)
                ans_ix                = options.index(choice)
                ans[ans_ix]           = np.abs(resp)
                ans[np.abs(ans_ix-1)] = -np.abs(resp)

                self.forced_choice.loc[0, [f'{options[0]}_v_{options[1]}_{options[0]}']] = ans[0]
                self.forced_choice.loc[0, [f'{options[0]}_v_{options[1]}_{options[1]}']] = ans[1]   
                self.forced_choice.loc[0, [f'{options[0]}_v_{options[1]}_reaction_time']] = rt

            self.forced_choice.index = [self.sub_id]
            self.forced_choice.columns = ['forced_choice_' + c for c in self.forced_choice.columns]
                
            return self.forced_choice

    # process iq
    # process misc qs


def parse_csv(file_path, snt_version='standard', out_dir=None):
    
    # directories
    if out_dir is None: 
        out_dir = Path(os.getcwd())
    if not os.path.exists(out_dir):
        print('Creating output directory')
        os.makedirs(out_dir)

    snt_dir = Path(f'{out_dir}/Organized')
    if not os.path.exists(snt_dir):
        print('Creating subdirectory for organized snt data')
        os.makedirs(snt_dir)

    post_dir = Path(f'{out_dir}/Posttask')
    if not os.path.exists(post_dir):
        print('Creating subdirectory for organized post task data')
        os.makedirs(post_dir)   

    # parse file
    parser = ParseCsv(file_path, snt_version=snt_version, verbose=0)
    snt, post = parser.run()
    snt.to_excel(Path(f'{snt_dir}/snt_{parser.sub_id}.xlsx'), index=False)
    post.to_excel(Path(f'{post_dir}/snt-posttask_{parser.sub_id}.xlsx'), index=True)


#------------------------------------------------------------------------------------------
# parse snt dots jpgs
#------------------------------------------------------------------------------------------


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
        erod_img   = sp.ndimage.binary_erosion(binary_img, iterations=3) # erode to get rid of specks
        recon_img  = sp.ndimage.binary_propagation(erod_img, mask=erod_img) * 1 # fill in 

        # segment image
        # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_spectral_clustering.html#sphx-glr-advanced-image-processing-auto-examples-plot-spectral-clustering-py
        # Convert the image into a graph with the value of the gradient on the edges
        graph = sk.feature_extraction.image.img_to_graph(binary_img, mask=recon_img.astype(bool))

        # Take a decreasing function of the gradient: we take it weakly
        # dependant from the gradient the segmentation is close to a voronoi
        graph.data = np.exp(-graph.data / graph.data.std())

        try: 
             # Force the solver to be arpack, since amg is numerically unstable
            labels   = sk.cluster.spectral_clustering(graph, n_clusters=4)
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


#------------------------------------------------------------------------------------------
# compute behavioral variables
#------------------------------------------------------------------------------------------


class ComputeBehavior:

    # TODO: can I clean some of this up by making some functions decorators?

    def __init__(self, file, weight_types=False, decision_types=False, coord_types=False):
    
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # fragmented df
        np.seterr(divide='ignore', invalid='ignore') # division by 0 in some of our operations
            
        #-------------
        # load in data
        #-------------
        
        if file is None: 

            self.file_path = None
            self.sub_id    = None   

        else:          

            if type(file) is not str: # eg for easy unittesting
                
                self.file_path = None
                self.sub_id    = None
                self.data      = copy.deepcopy(file)
            
            else: 
                
                self.file_path = Path(file)
                self.sub_id    = self.file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
                
                # if not utils.is_numeric(self.sub_id): 
                #     raise Exception('Subject id isnt numeric; filename should have pattern: "snt_subid*.xlsx"')

                if self.file_path.suffix == '.xlsx':  self.data = copy.deepcopy(pd.read_excel(self.file_path, engine='openpyxl'))
                elif self.file_path.suffix == '.xls': self.data = copy.deepcopy(pd.read_excel(self.file_path))
                elif self.file_path.suffix == '.csv': self.data = copy.deepcopy(pd.read_csv(self.file_path))
                else: raise Exception(f'File type {self.file_path.suffix} not recognized')
    
                self.check_input_shape(self.data, (63, self.data.shape[1])) # should have 63 trials
 
            #---------------
            # clean up input
            #---------------
            
            # want decisions in 2d
            if 'affil' not in self.data.columns: # backward compatability
                self.data['decision'] = self.data['decision'].astype(int)
                dim_mask  = np.vstack([(self.data['dimension'] == 'affil').values, 
                                    (self.data['dimension'] == 'power').values]).T
                self.data[['affil', 'power']] = self.data['decision'].values[:, np.newaxis] * (dim_mask * 1)
                            
            # data types
            type_dict = {'decision_num': int, 'scene_num': int, 'dimension': object,
                        'char_role_num': int, 'char_decision_num': int,
                        'button_press': int, 'decision': int, 'affil': int, 'power': int,
                        'reaction_time': float, 'onset': float}
            for col in self.data: 
                if self.data[col].dtype != type_dict[col]:
                    self.data[col] = self.data[col].astype(type_dict[col])

        #----------------
        # what to compute
        #----------------
        
        # character list
        self.characters = {'first':     {'role_num':1, 'behavior':pd.DataFrame()},
                           'second':    {'role_num':2, 'behavior':pd.DataFrame()},
                           'assistant': {'role_num':3, 'behavior':pd.DataFrame()},
                           'powerful':  {'role_num':4, 'behavior':pd.DataFrame()},
                           'boss':      {'role_num':5, 'behavior':pd.DataFrame()},
                           'neutral':   {'role_num':9, 'behavior':pd.DataFrame()}}
        
        # decision types: current v. previous
        self.decision_types = {'': self.current_decisions, '_prev': self.previous_decisions}
        if decision_types is False:  dts = ['']
        elif decision_types is True: dts = list(self.decision_types.keys())
        else:                        dts = decision_types
        self.decision_types = {k: self.decision_types[k] for k in dts}
        
        # weighting types: constant, linear decay v. exponential decay
        self.weight_types = {'':'constant', '_linear-decay':'linear', '_exponential-decay':'exponential'}
        if weight_types is False:  wts = ['']
        elif weight_types is True: wts = list(self.weight_types.keys())
        else:                      wts = weight_types
        self.weight_types = {k: self.weight_types[k] for k in wts}
        
        # coordinate types: actual v. counterfactual
        self.coord_types = {'':self.actual_coords, '_cf':self.counterfactual_coords}
        if coord_types is False:  cts = ['']
        elif coord_types is True: cts = list(self.coord_types.keys())
        else:                     cts = coord_types
        self.coord_types = {k: self.coord_types[k] for k in cts}
        

    #--------
    # helpers
    #--------


    def check_input_shape(self, input_, exp_shapes):
        ''' at various points: input to class as well as character specific subsets '''
        if type(exp_shapes) != list: exp_shapes = [exp_shapes]
        matches = np.sum([input_.shape == e for e in exp_shapes])
        if matches == 0:
            str_ = (' ').join([f'({e[0]},{e[1]})' for e in exp_shapes])
            raise Exception(f'Shape mismatch: {input_.shape}!= any of expected shapes: {str_}')
        else:
            return True
        
        
    def make_polygon(self, coords):
        
        shape    = sp.spatial.ConvexHull(coords)
        vertices = coords[shape.vertices]
        polygon  = geometry.Polygon(vertices)
        return polygon
    
    
    def cumulative_sum(self, values, axis=0):
        
        ''' mainly just to be able to pass a function like for prev decision etc'''
        return np.cumsum(values, axis=axis).astype(float)

    
    def cumulative_mean(self, values, counts):
        
        ''' divide sum at each time point by the response count '''
        cum_sum   = self.cumulative_sum(values)
        cum_count = self.cumulative_sum(counts)
        return [cum_sum / cum_count, (cum_sum, cum_count)]       
    
    
    #---------------------------------------
    # (1) get decisions: current or previous
    #---------------------------------------
    
    
    def current_decisions(self, decisions=None):
        
        if decisions is None: decisions = self._decisions_raw
        return decisions

        
    def previous_decisions(self, decisions=None, shift_by=1, replace_with=0):
        
        ''' if on each trial the subj represents the last chosen decision/coordinates ''' 
        
        if decisions is None: decisions = self._decisions_raw
        decisions_prev            = np.ones_like(decisions) * replace_with 
        decisions_prev[shift_by:] = np.array(decisions)[0:-shift_by]
        return decisions_prev
        

    #---------------------------------------------------------------
    # (2) weight decisions: constant, linear decay, expnential decay
    #---------------------------------------------------------------
    
    
    def weight_decisions(self, decisions=None, decay='constant'):
        
        ''' different ways to weight the decisions '''
        
        if decisions is None: decisions = self._decisions
        n_trials = len(decisions)
        if decay == 'constant':      weights = np.ones(n_trials)[:,None]
        elif decay == 'linear':      weights = utils.linear_decay(1, 1/n_trials, n_trials)[:,None]
        elif decay == 'exponential': weights = utils.exponential_decay(1, 1/n_trials, n_trials)[:,None]
        return decisions * weights

        
    #----------------------------------------------------
    # (3) calculate coordinates: actual or counterfactual
    #----------------------------------------------------


    def actual_coords(self, decisions=None):
        
        ''' actual coords '''
        if decisions is None: decisions = self._decisions
        return self.cumulative_sum(decisions).astype(float)
    
    
    def counterfactual_coords(self, decisions=None):
        
        ''' counterfactual ('what if') coordinates wrt on each trial '''
        if decisions is None: decisions = self._decisions
        
        return (self.cumulative_sum(decisions) - (2*decisions)).astype(float)
        
        
    #---------------
    # main functions
    #---------------
    
    
    def simulate_consistent_decisions(self, decisions=None):
        
        ''' perfectly consistent and perfectly inconsistent decisions & coordinates
            - incrementally accounts for non-responses
            - incremental mean: accumulated choices / number of choices made, at each time point
        '''
        
        if decisions is None: decisions = self._decisions
        
        # want the cumulative mean to control for non-responses: divide by num of responses to that point    
        resp_mask = abs(decisions) 

        # most & least consistency possible [for decision pattern]
        con_decs   = resp_mask * 1
        incon_decs = np.zeros_like(resp_mask)
        for n_dim in range(2): 
            dim_mask  = resp_mask[:, n_dim] != 0 
            con_decs_ = con_decs[dim_mask, n_dim] # if its not 0, its a response
            incon_decs[dim_mask, n_dim] = [n if not i % 2 else -n for i,n in enumerate(con_decs_)] # flip every other sign

        return [incon_decs, con_decs, resp_mask]

    
    def cumulative_consistency(self, decisions=None):

        if decisions is None: decisions = self._decisions
        
        # simulate possible behavior
        [incon_decs, con_decs, resp_mask] = self.simulate_consistent_decisions(decisions)  
        min_coords = self.cumulative_sum(incon_decs) / self.cumulative_sum(resp_mask) # adjust for response counts at each time point
        max_coords = self.cumulative_sum(con_decs) / self.cumulative_sum(resp_mask)

        # 1d consistency = abs value coordinate, scaled by min and max possible coordinate  
        cum_mean, _    = self.cumulative_mean(decisions, resp_mask)
        consistency_1d = (np.abs(cum_mean) - min_coords) / (max_coords - min_coords) # min max scaled

        # 2d consistency = decision vector length, scaled by min and max possible vector lengths
        min_r, max_r  = (np.array([np.linalg.norm(v) for v in min_coords]), np.array([np.linalg.norm(v) for v in max_coords]))
        cum_mean_r    = np.array([np.linalg.norm(v) for v in cum_mean])
        consistency_r = (cum_mean_r - min_r) / (max_r - min_r)

        # return both dimensions separately & 2d
        return np.hstack([consistency_1d, consistency_r[:, np.newaxis]])

    
    def add_3rd_dimension(self, U, V, ori):
        
        ''' add 3rd dimension to U & V, as well as to the origin coordinates
            - U is vector of interest, z-axis will vary w/ number of interactions
            - ori will be subtracted from U, so its z-axis will also vary w/ number of interactions
            - V is reference vector, z-axis will remain fixed
        '''
        if V.ndim == 2:   V = V[0]
        if ori.ndim == 1: ori = ori[np.newaxis]

        num = np.arange(1, len(U) + 1)[:,np.newaxis]
        U   = np.concatenate([U, num], axis=1) # changes w/ num of interactions
        V   = np.repeat(np.hstack([V, len(U)])[np.newaxis], len(U), axis=0) # fixed
        ori = np.concatenate([np.repeat(ori, len(U), axis=0), num], axis=1) # changes w/ num of interactions   

        return U, V, ori                           
    
    
    def calculate_polar(self, ref_frame, n_dim=2):
        
        ''' 
            calculate directional angles between (ori to poi) and (ori to ref) [optional 3rd dimension]
            - origin (ori)
            --- neu: (0, 0, [interaction # (1:12)]) - note that 'origin' moves w/ interactions if in 3d
            --- pov: (6, 0, [interaction # (1:12)])
            - reference vector (ref)
            --- neu: (6, 0, [max interaction (12)])
            --- pov: (6, 6, [max interaction (12)])
            - point of interaction vector (poi): (curr. affil coord, power coord, [interaction # (1:12)])
            to get directional vetctors (poi-ori), (ref-ori)
        '''
    
        poi = self._coords
        ref, ori, drn = ref_frame['ref'], ref_frame['ori'], ref_frame['dir']

        # if 3d add in 3rd dimension
        if n_dim == 3: 
            poi, ref, ori = self.add_3rd_dimension(poi, ref[0], ori)
            drn = None # may not be correct for neutral origin... not sure yet

        angles    = utils.calculate_angle(poi-ori, ref-ori, force_pairwise=False, direction=drn)
        distances = [np.linalg.norm(v) for v in poi-ori] # l2 norm is euclidean distance from ori
        return [angles, distances]

    
    def quadrant_overlap(self, coords):
        
        ''' the percentage overlap of the decision polygon (made by vertices of coordinates) with each of the 4 2D quadrants 
            - sum == 100
        '''
        
        quadrants = {'1': np.array([[0,0], [0,6],  [6,0],  [6,6]]),   
                     '2': np.array([[0,0], [0,6],  [-6,0], [-6,6]]),
                     '3': np.array([[0,0], [0,-6], [-6,0], [-6,-6]]), 
                     '4': np.array([[0,0], [0,-6], [6,0],  [6,-6]])}
        try: 
            polygon = self.make_polygon(coords)            
            return [polygon.intersection(self.make_polygon(qC)).area/polygon.area for q, qC in quadrants.items()]
        except: # might not be enough datapoints yet
            return [np.nan, np.nan, np.nan, np.nan]
        
        
    #----------
    # run stuff
    #----------
    
    
    def run(self):
        
        self.behavior = pd.DataFrame()
        
        # diff types of assumptins about space
        self._types = [[dt,wt,ct] for dt in list(self.decision_types.keys()) for wt in list(self.weight_types.keys()) for ct in list(self.coord_types.keys())]
        for self._dt, self._wt, self._ct in self._types:
            
            # diff characters
            char_dfs = []
            for self._character in self.characters.keys(): 
                self.within_character()
                char_dfs.append(self.characters[self._character]['behavior'])
            
            # combine
            char_df = pd.concat(char_dfs, axis=0).sort_index() 
            char_df.reset_index(inplace=True, drop=True)
            self.behavior = pd.concat([self.behavior, char_df], axis=1) # add as columns
            self.behavior['char_decision_num'] = self.data['char_decision_num']
            
            # all characters
            self.across_characters()
            self.behavior.reset_index(inplace=True, drop=True)

        # add in extra info columns
        info = self.data[[c for c in self.data.columns if c not in ['affil','power']]]
        info.reset_index(inplace=True, drop=True)
        self.behavior.reset_index(inplace=True, drop=True)
        self.behavior = pd.concat([info, self.behavior], axis=1)

    
    def within_character(self):
        
        ''' compute variables w/in character '''
        
        suffix = f'{self._dt}{self._wt}{self._ct}'
        
        role_num, character_behavior = self.characters[self._character]['role_num'], self.characters[self._character]['behavior']
        
        # indices & masks for dimensions & responses
        self._ixs = np.where(self.data['char_role_num'] == role_num)[0] # df indices for this character in
        data      = self.data.loc[self._ixs, ['dimension', 'button_press', 'affil', 'power']]
        dim_mask  = np.vstack([(data['dimension'] == 'affil').values, (data['dimension'] == 'power').values]).T 
        resp_mask = np.multiply(dim_mask, (data['button_press'] != 0).values[:, np.newaxis]) # dont count non-responses for averages, etc
        
        # get decisions
        self._decisions_raw = data[['affil', 'power']].values
        self.check_input_shape(self._decisions_raw, [(12,2), (3,2)])
        self.check_input_shape(self._decisions_raw, resp_mask.shape)
        self._decisions = self.decision_types[self._dt]()
        
        # weight decisions & get coordinates
        self._decisions = self.weight_decisions(decay=self.weight_types[self._wt]) 
        self._coords    = self.coord_types[self._ct]()
        
        character_behavior[[f'affil{suffix}',       f'power{suffix}']]        = self._decisions
        character_behavior[[f'affil_coord{suffix}', f'power_coord{suffix}']]  = self._coords

        # calculate cumulative mean along each dimension [-1,+1]
        character_behavior[[f'affil_mean{suffix}', f'power_mean{suffix}']], _ = self.cumulative_mean(self._decisions, resp_mask)

        # calculate culumative consistency: mean of absolute value/vector length along each dimension [0,1]
        character_behavior[[f'affil_consistency{suffix}', f'power_consistency{suffix}',f'consistency{suffix}']] = self.cumulative_consistency() # (affil, power, 2d)

        # calculate angles & distances
        ref_frames = {'neu': {'ori': np.array([[0,0]]), 'ref': np.array([[6,0]]), 'dir': False},
                      'pov': {'ori': np.array([[6,0]]), 'ref': np.array([[6,6]]), 'dir': None}} 
        for n_dim in [2, 3]:    
            for ot in ['neu', 'pov']:
                angles, distances = self.calculate_polar(ref_frames[ot], n_dim=n_dim)                
                character_behavior[f'{ot}{n_dim}d_angle{suffix}'] = angles
                character_behavior[f'{ot}_distance{suffix}']      = distances

        character_behavior.index = [self._ixs]
        
        
    def across_characters(self):
        
        '''  variables across characters '''

        suffix = f'{self._dt}{self._wt}{self._ct}'
        coords = self.behavior.loc[:, [f'affil_coord{suffix}', f'power_coord{suffix}', 'char_decision_num']].values

        for t_num in range(1, 64):

            # in 2d
            overlap = self.quadrant_overlap(coords[:t_num, 0:2])
            try: 
                shape = sp.spatial.ConvexHull(coords[:t_num, 0:2]) # outputs perim & area when 2D
                shape = {'perim': shape.area, 'area': shape.volume}
            except: 
                shape = {'perim': np.nan, 'area': np.nan} 

            # in 3d
            
            try:    shape['volume'] = sp.spatial.ConvexHull(coords[:t_num, :]).volume
            except: shape['volume'] = np.nan

            self.behavior.loc[t_num-1, f'perimeter{suffix}']  = shape['perim']
            self.behavior.loc[t_num-1, f'area{suffix}']       = shape['area']
            self.behavior.loc[t_num-1, f'volume{suffix}']     = shape['volume']
            self.behavior.loc[t_num-1, f'Q1_overlap{suffix}'] = overlap[0]
            self.behavior.loc[t_num-1, f'Q2_overlap{suffix}'] = overlap[1]
            self.behavior.loc[t_num-1, f'Q3_overlap{suffix}'] = overlap[2]
            self.behavior.loc[t_num-1, f'Q4_overlap{suffix}'] = overlap[3]


def compute_behavior(file_path, out_dir=None):

    # directories
    if out_dir is None: 
        out_dir = Path(f'{os.getcwd()}/Behavior')
    else:
        if '/Behavior' not in out_dir: 
            out_dir = Path(f'{out_dir}/Behavior')
    if out_dir is not False and not os.path.exists(out_dir):
        print('Creating subdirectory for behavior')
        os.makedirs(out_dir)
    
    # compute behavior & output
    computer  = ComputeBehavior(file=file_path) # leave defaults for now:
    computer.run()
    sub_id = Path(file_path).stem.split('_')[1]
    computer.behavior.to_excel(Path(f'{out_dir}/snt_{sub_id}_behavior.xlsx'), index=False)


def summarize_behavior(file_paths, out_dir=None):
    
    if out_dir is None: 
        out_dir = Path(f'{os.getcwd()}/preprocessed_behavior')
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    summaries = []

    file_paths = sorted((f for f in file_paths if (not f.startswith(".")) & ("~$" not in f)), key=str.lower) # ignore hidden files & sort alphabetically
    for s, file_path in enumerate(file_paths):
        print(f'Summarizing {s+1} of {len(file_paths)}', end='\r')

        ### load in data ###
        file_path = Path(file_path)
        if file_path.suffix == '.xlsx':  behavior = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.suffix == '.xls': behavior = pd.read_excel(file_path)
        elif file_path.suffix == '.csv': behavior = pd.read_csv(file_path)

        sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'

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
            else:                  summary.loc[0, col + '_mean'] = pycircstat.mean(behavior[col])

        # last trial only
        end_df = pd.DataFrame(behavior.loc[62,cols].values).T
        end_df.columns = [c + '_end' for c in cols]

        summary = pd.concat([summary, end_df], axis=1)

        summaries.append(summary)

    summary = pd.concat(summaries)
    summary.to_excel(Path(f'{out_dir}/SNT-behavior_n{summary.shape[0]}.xlsx'), index=False)


#------------------------------------------------------------------------------------------
# compute mvpa stuff
#------------------------------------------------------------------------------------------


def get_rdv_trials(trial_ixs, rdm_size=63):

    # fill up a dummy rdm with the rdm ixs
    rdm  = np.zeros((rdm_size, rdm_size))
    rdm_ixs = utils.combos(trial_ixs, k=2)
    for i in rdm_ixs: 
        rdm[i[0],i[1]] = 1
        rdm[i[1],i[0]] = 1
    rdv = utils.symm_mat_to_ut_vec(rdm)
    
    return (rdv == 1), np.where(rdv==1)[0] # boolean mask, ixs


def get_char_rdv(char_int, trial_ixs=None, rdv_to_mask=None):
    ''' gets a categorical rdv for a given character (represented as integers from 1-5)
        should make more flexible to also be able to grab 
        NOTE: this is the upper triangle
    '''
    
    if trial_ixs is not None:
        decisions = info.decision_trials.loc[trial_ixs,:].copy()        
    else:
        decisions = info.decision_trials
    
    # char_rdm = np.ones((decisions.shape[0], decisions.shape[0]))
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
        decisions = info.decision_trials.loc[trial_ixs,:]
    else:
        decisions = info.decision_trials
    cols = []
    
    # time-related drift rdms - continuous-ish
    time_rdvs = np.vstack([utils.ut_vec_pw_dist(np.array(decisions['cogent_onset'])) ** p for p in range(1,8)]).T
    cols = cols + [f'time{t+1}' for t in range(time_rdvs.shape[1])]

    # narrative rdms - continuous-ish
    narr_rdvs = np.vstack([utils.ut_vec_pw_dist(decisions[col].values) for col in ['slide_num','scene_num','char_decision_num']]).T
    cols = cols + ['slide','scene','familiarity']

    # dimension rdms - categorical 
    dim_rdv = utils.ut_vec_pw_dist(np.array((decisions['dimension'] == 'affil') * 1).reshape(-1,1), metric=metric) # diff or same dims?
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
        out_dir = Path(f'{os.getcwd()}/RDVs')
    else:
        if '/RDVs' not in out_dir: 
            out_dir = Path(f'{out_dir}/RDVs')
    if not os.path.exists(out_dir):
        print('Creating subdirectory for RDVs')
        os.makedirs(out_dir)

    ### load in data ###
    file_path = Path(file_path)
    sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
    assert utils.is_numeric(sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

    file_path = Path(file_path)
    if file_path.suffix == '.xlsx':  behavior_ = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.suffix == '.xls': behavior_ = pd.read_excel(file_path)
    elif file_path.suffix == '.csv': behavior_ = pd.read_csv(file_path)

    # output all the decision type models?
    if output_all: 
        suffixes = utils.flatten_nested_lists([[f'{wt}{dt}' for dt in ['','_prev','_cf'] for wt in ['', '_linear-decay', '_expon-decay']]]) 
    else: 
        suffixes = ''
        
    for sx in suffixes: 

        behavior     = behavior_[['decision', 'reaction_time', 'button_press', 'char_decision_num', 'char_role_num',f'affil{sx}',f'power{sx}',f'affil_coord{sx}',f'power_coord{sx}']]
        end_behavior = behavior[behavior['char_decision_num'] == 12].sort_values(by='char_role_num')

        for outname, behav in {sx: behavior, f'{sx}_end': end_behavior}.items(): 

            decisions = np.sum(behav[[f'affil{sx}',f'power{sx}']],1)
            coords    = behav[[f'affil_coord{sx}',f'power_coord{sx}']].values

            rdvs = get_ctl_rdvs(trial_ixs=behav.index)
            rdvs.loc[:,'reaction_time'] = utils.ut_vec_pw_dist(np.nan_to_num(behav['reaction_time'], 0))
            rdvs.loc[:,'button_press']  = utils.ut_vec_pw_dist(np.array(behav['button_press']))

            ######################################################
            # relative distances between locations
            # - can try other distances: e.g., manhattan which would be path distance
            ######################################################

            metric = 'euclidean'
            rdvs.loc[:,'place_2d']       = utils.ut_vec_pw_dist(coords, metric=metric)
            rdvs.loc[:,'place_affil']    = utils.ut_vec_pw_dist(coords[:,0], metric=metric)
            rdvs.loc[:,'place_power']    = utils.ut_vec_pw_dist(coords[:,1], metric=metric)
            rdvs.loc[:,'place_positive'] = utils.ut_vec_pw_dist(np.sum(coords, 1), metric=metric)

            #     # newer adds:
            #     rdvs['place_2d_scaled', utils.ut_vec_pw_dist(behavior[['affil_coord_scaled', 'power_coord_scaled']])) # dont zscore cuz already scaled
            #     rdvs['place_2d_exp_decay', utils.ut_vec_pw_dist(behavior[['affil_coord_exp-decay', 'power_coord_exp-decay']]))
            #     rdvs['place_2d_exp_decay_scaled', utils.ut_vec_pw_dist(behavior[['affil_coord_exp-decay_scaled', 'power_coord_exp-decay_scaled']]))

            ######################################################
            # distances from ref points (poi - ref)
            # -- ori to poi vector (poi - [0,0]) 
            # -- pov to poi vector (poi - [6,0]) 
            ######################################################

            for origin, ori in {'neu':[0,0], 'pov':[6,0]}.items():

                V = coords - ori

                rdvs.loc[:,f'{metric}_distance_{origin}'] = utils.ut_vec_pw_dist(np.array([np.linalg.norm(v) for v in V]), metric=metric)
                rdvs.loc[:,f'angular_distance_{origin}']  = utils.symm_mat_to_ut_vec(utils.angular_distance(V)) 
                rdvs.loc[:,f'cosine_distance_{origin}']   = utils.symm_mat_to_ut_vec(utils.cosine_distance(V))

            ######################################################
            # others
            ######################################################

            # decision directon: +1 or -1
            direction_rdv = utils.ut_vec_pw_dist(behav['decision'].values.reshape(-1,1))
            direction_rdv[direction_rdv > 1] = 1 
            rdvs.loc[:,'decision_direction'] = direction_rdv

            # output
            rdvs.to_excel(Path(f'{out_dir}/snt_{sub_id}{outname}_rdvs.xlsx'), index=False)
  

#------------------------------------------------------------------------------------------
# helpers
#------------------------------------------------------------------------------------------


def fake_data():

    # make better to test what certain patterns would look like in the behavior
    dimension = np.array([['affil', 'affil', 'affil', 'power', 'affil', 'power', 'power',
                        'affil', 'affil', 'power', 'affil', 'power', 'power', 'power',
                        'neutral', 'neutral', 'affil', 'power', 'affil', 'power', 'power',
                        'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                        'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                        'neutral', 'power', 'power', 'affil', 'affil', 'affil', 'affil',
                        'power', 'affil', 'affil', 'power', 'affil', 'affil', 'affil',
                        'affil', 'affil', 'power', 'power', 'power', 'affil', 'power',
                        'affil', 'affil', 'power', 'power', 'power', 'power', 'affil']]).reshape(-1,1)
    char_role_nums = np.array([[1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 4, 2, 1, 1, 9, 9, 1, 2, 4, 2, 4, 4,
                                4, 4, 1, 2, 2, 2, 1, 5, 5, 3, 5, 3, 3, 9, 3, 3, 3, 3, 5, 5, 5, 5,
                                5, 5, 2, 2, 3, 2, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3]]).reshape(-1,1)
    char_dec_nums = np.array([[1,2, 3, 4, 1, 2, 5, 6, 7,  3,  1,  4,  8,  9,  1,  2, 10,
                                5,2, 6, 3, 4, 5, 6, 11,  7,  8,  9, 12,  1,  2,  1,  3,  2,
                                3,3, 4, 5, 6, 7, 4, 5, 6,  7,  8,  9, 10, 11,  8, 12, 10,
                                11,12, 7, 8, 9, 10, 11, 12,  9, 10, 11, 12]]).reshape(-1,1)
    # fake decisions
    button_press = np.array([np.random.choice([1,2 ]) for _ in range(63)]).reshape(-1,1)
    decisions    = np.array([np.random.choice([-1,1]) for _ in range(63)]).reshape(-1,1)
    fake_data = pd.DataFrame(np.hstack([dimension, char_role_nums, char_dec_nums, button_press, decisions]), 
                                 columns=['dimension', 'char_role_num', 'char_decision_num', 'button_press', 'decision'])
    return fake_data

