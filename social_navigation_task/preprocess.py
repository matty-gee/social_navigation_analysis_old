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

import info 
import utils

pkg_dir = str(Path(__file__).parent.absolute())

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
        self.sub_id     = self.data.prolific_id[0]
        self.task_ver   = self.data['task_ver'].values[0]
        self.snt_ver    = snt_version

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
            race_cols = utils.find_pattern(self.data.columns, 'race_*_*')
            rename = {}
            for col in race_cols:
                split_ = col.split('_')
                rename[col] = f'judgment_{split_[1]}_{split_[0]}_{split_[2]}'
            self.data.rename(columns=rename, inplace=True)
            
        return self.data
   
    def run_pipeline(self):

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
            validated_decisions = validated_decisions[self.experiment]

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
            self.snt['reaction_time'] = snt_rts[np.array(validated_decisions['Slide_num']) - 1] / 1000
        
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
            if self.snt_ver.isin(['standard', 'schema']):
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


def parse_csv(file_path, out_dir=None):
    
    # directories
    if out_dir is None: out_dir = Path(os.getcwd())
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
    parser = ParseCsv(file_path, verbose=0)
    snt, post = parser.run_pipeline()
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

# def compute_behavior(file_path, out_dir=None):
    
#     # annoying warnings that I dont think matter really
#     from warnings import simplefilter
#     simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # fragmented df
#     np.seterr(divide='ignore', invalid='ignore') # division by 0 in some of our operations

#     # out directory
#     if out_dir is None: 
#         out_dir = Path(f'{os.getcwd()}/Behavior')
#     else:
#         if '/Behavior' not in out_dir: 
#             out_dir = Path(f'{out_dir}/Behavior')
#     if not os.path.exists(out_dir):
#         print('Creating subdirectory for behavior')
#         os.makedirs(out_dir)

#     ### load in data ###
#     file_path = Path(file_path)
#     sub_id = file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'

#     if not utils.is_numeric(sub_id): 
#         raise NameError('Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"')

#     file_path = Path(file_path)
#     if file_path.suffix == '.xlsx':  behavior = pd.read_excel(file_path, engine='openpyxl')
#     elif file_path.suffix == '.xls': behavior = pd.read_excel(file_path)
#     elif file_path.suffix == '.csv': behavior = pd.read_csv(file_path)


#     #####################################
#     ### compute things w/in character ###
#     #####################################

#     for c in [1,2,3,4,5,9]:

#         mask_df = (behavior['char_role_num']==c).values
#         ixs = np.where(behavior['char_role_num']==c)[0]
#         behav = behavior.loc[ixs,:]
#         decisions = behav[['affil','power']].values

#         # dont count non-responses for averages, etc
#         responded  = (behav['button_press'] != 0).values.reshape(-1,1)
#         mask_affil = (behav['dimension'] == 'affil').values
#         mask_power = (behav['dimension'] == 'power').values
#         resp_mask  =  np.vstack([mask_affil, mask_power]).T * responded

#         # if responded

#         ################################
#         ###  decisions & coordinates ###
#         ################################

#         # different decision weighting schemes
#         nt = len(decisions) 
#         constant    = np.ones(nt)[:,None] # standard weighting with a constant
#         linear      = utils.linear_decay(1, 1/nt, nt)[:,None]
#         exponential = utils.exponential_decay(1, 1/nt, nt)[:,None]
        
#         for wt, w in {'':constant, '_linear-decay':linear, '_expon-decay':exponential}.items():

#             ### weight the decisions & make coordinates 
#             decisions_w = decisions * w
#             coords_w = np.cumsum(decisions_w, 0)

#             ### current decisions & coordinates
#             # - eg, if on each trial the subj represents the the decision/coordinates they chose
#             behavior.loc[ixs, [f'affil{wt}', f'power{wt}']] = decisions_w.astype(float)
#             behavior.loc[ixs, [f'affil_coord{wt}', f'power_coord{wt}']] = coords_w.astype(float)

#             ### previous decisions & coordinates
#             # - eg, if on each trial the subj represents the last chosen decision/coordinates
#             shifted_decisions_w = _shift_down(decisions_w, by=1) 
#             behavior.loc[ixs, [f'affil{wt}_prev', f'power{wt}_prev']] = shifted_decisions_w
#             behavior.loc[ixs, [f'affil_coord{wt}_prev', f'power_coord{wt}_prev']] = np.cumsum(shifted_decisions_w, 0)

#             ### countefactual decisions & coordinates
#             # - counterfactual wrt each trial
#             # -- eg, if on each trial the subj represents the [ultimately] non-chosen decision/coordinates
#             prev_coords_w = coords_w - decisions_w # for each trial, undo the decisions by subtracting to get the prev coords
#             decisions_w_cf = -decisions_w # opposite decision
#             behavior.loc[ixs, [f'affil{wt}_cf',f'power{wt}_cf']] = decisions_w_cf
#             behavior.loc[ixs, [f'affil_coord{wt}_cf',f'power_coord{wt}_cf']] = prev_coords_w + decisions_w_cf # flip the decision made, and add back in 


#             ###########################
#             ### geometric variables ###
#             ###########################

#             # loop over each decision type
#             for dt in ['','_prev','_cf']:

#                 decisions = behavior.loc[ixs, [f'affil{wt}{dt}',f'power{wt}{dt}']].values
#                 coords    = behavior.loc[ixs, [f'affil_coord{wt}{dt}',f'power_coord{wt}{dt}']].values

#                 ### cumulative mean along each dimension [-1,+1] ###
#                 cum_resp = np.cumsum(resp_mask, 0)
#                 cum_mean = coords / cum_resp # divide each time point by the response count 
#                 if c != 9: # neu will get all nans
#                     assert np.all(cum_mean[-1,:] == coords[-1,:] / cum_resp[-1,:]), 'cumulative mean is off'

#                 behavior.loc[ixs, [f'affil_mean{wt}{dt}', f'power_mean{wt}{dt}']] = cum_mean

#                 ### culumative consistency: how far did they go in the direction they were traveling? [0,1] ###

#                 minmax = minmax_coords(decisions)

#                 # 1d consistency = abs value coordinate, scaled by min and max possible coordinate            
#                 behavior.loc[ixs, [f'affil_consistency{wt}{dt}', f'power_consistency{wt}{dt}']] = (np.abs(cum_mean) - minmax[0]) / (minmax[1] - minmax[0])

#                 # 2d consistency = decision vector length, scaled by min and max possible vector lengths
#                 min_r = np.array([np.linalg.norm(v) for v in minmax[0]])
#                 max_r = np.array([np.linalg.norm(v) for v in minmax[1]])
#                 r = np.array([np.linalg.norm(v) for v in cum_mean])
#                 behavior.loc[ixs, f'consistency{wt}{dt}'] = (r - min_r) / (max_r - min_r)

#                 ### multi-dimensional: angles & distances ###

#                 # directional angles between (ori to poi) and (ori to ref) [optional 3rd dimension]
#                 # - origin (ori)
#                 # --- neu: (0, 0, [interaction # (1:12)]) - note that 'origin' moves w/ interactions if in 3d
#                 # --- pov: (6, 0, [interaction # (1:12)])
#                 # - reference vector (ref)
#                 # --- neu: (6, 0, [max interaction (12)])
#                 # --- pov: (6, 6, [max interaction (12)])
#                 # - point of interaction vector (poi): (curr. affil coord, power coord, [interaction # (1:12)])
#                 # to get directional vetctors (poi-ori), (ref-ori)

#                 ref_frames = {'neu': {'ori': np.array([[0,0]]), 'ref': np.array([[6,0]]), 'dir': False},
#                               'pov': {'ori': np.array([[6,0]]), 'ref': np.array([[6,6]]), 'dir': None}} 
#                 int_num    = np.arange(1, len(behav)+1).reshape(-1, 1)

#                 for ndim in np.arange(2, 4):    
#                     for ot in ['neu', 'pov']:

#                         ## angles between ori-poi & ori-ref
#                         ref_frame = ref_frames[ot]
#                         poi = coords
#                         ref = ref_frame['ref']
#                         ori = ref_frame['ori']
#                         drn = ref_frame['dir'] # direction for angle calc

#                         # if 3d, add 3rd dimension
#                         if ndim == 3: 
#                             poi = np.concatenate([poi, int_num], 1) # changes w/ num of interactions
#                             ref = np.repeat(np.hstack([ref[0], 12]).reshape(1, -1), len(poi), 0) # fixed
#                             ori = np.concatenate([np.repeat(ori.reshape(1, -1), len(poi), 0), int_num], 1) # changes w/ num of interactions
#                             drn = None # may not be correct for neutral origin... not sure yet

#                         # account for origin
#                         poi = poi - ori
#                         ref = ref - ori

#                         # angle between ori-poi & ori-ref
#                         behavior.loc[ixs, f'{ot}{ndim}d_angle{wt}{dt}'] = utils.calculate_angle(poi, ref, direction=drn) # outputs radians

#                         ## distances from ori-poi
#                         if ndim == 2: 
#                             behavior.loc[ixs, f'{ot}_distance{wt}{dt}'] = [np.linalg.norm(v) for v in poi] # l2 norm is euclidean distance from ori

#     ######################################
#     ## variables across all characters ###
#     ######################################

#     suffixes = utils.flatten_nested_lists([[f'{wt}{dt}' for dt in ['','_prev','_cf'] for wt in ['', '_linear-decay', '_expon-decay']]])
    
#     for sx in suffixes:

#         quadrants = {'1': np.array([[0,0], [0,6],  [6,0],  [6,6]]),   '2': np.array([[0,0], [0,6],  [-6,0], [-6,6]]),
#                      '3': np.array([[0,0], [0,-6], [-6,0], [-6,-6]]), '4': np.array([[0,0], [0,-6], [6,0],  [6,-6]])}

#         for t in range(0, 63):

#             # in 3d
#             X = behavior.loc[:t, [f'affil_coord{sx}', f'power_coord{sx}', 'char_decision_num']].values

#             try:    vol = sp.spatial.ConvexHull(X).volume
#             except: vol = np.nan

#             # in 2d
#             try:    

#                 shape = sp.spatial.ConvexHull(X[:,0:2])
#                 perim = shape.area # perimeter
#                 area  = shape.volume  # area when 2D
#                 poly  = geometry.Polygon(X[:,0:2][shape.vertices]) 
                
#                 overlap = []
#                 for q, C in quadrants.items(): # overlap w/ quadrants
#                     Q = geometry.Polygon(C[sp.spatial.ConvexHull(C).vertices])
#                     overlap.append(poly.intersection(Q).area/poly.area) 
#             except: 

#                 perim = np.nan
#                 area  = np.nan
#                 overlap = [np.nan,np.nan,np.nan,np.nan]

#             behavior.loc[t, f'perimeter{sx}']  = perim
#             behavior.loc[t, f'area{sx}']       = area
#             behavior.loc[t, f'volume{sx}']     = vol
#             behavior.loc[t, f'Q1_overlap{sx}'] = overlap[0]
#             behavior.loc[t, f'Q2_overlap{sx}'] = overlap[1]
#             behavior.loc[t, f'Q3_overlap{sx}'] = overlap[2]
#             behavior.loc[t, f'Q4_overlap{sx}'] = overlap[3]
            
            
#     behavior.to_excel(Path(f'{out_dir}/snt_{sub_id}_behavior.xlsx'), index=False)


class ComputeBehavior:

    def __init__(self, file_path, weight_types=None, decision_types=None, out_dir=None):
    
        # annoying warnings
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # fragmented df
        np.seterr(divide='ignore', invalid='ignore') # division by 0 in some of our operations

        if file_path is None: # easy unittesting
            self.file_path = None
            self.sub_id = None

        else: 

            # load in data
            self.file_path = Path(file_path)
            self.sub_id    = self.file_path.stem.split('_')[1] # expects a filename like 'snt_subid_*'
            assert utils.is_numeric(self.sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

            if self.file_path.suffix == '.xlsx':  self.behavior = pd.read_excel(self.file_path, engine='openpyxl')
            elif self.file_path.suffix == '.xls': self.behavior = pd.read_excel(self.file_path)
            elif self.file_path.suffix == '.csv': self.behavior = pd.read_csv(self.file_path)

            # out directory
            if out_dir is False: 
                self.out_dir = False
            elif out_dir is None: 
                self.out_dir = Path(f'{os.getcwd()}/Behavior')
            else:
                if '/Behavior' not in out_dir: 
                    self.out_dir = Path(f'{out_dir}/Behavior')
            if self.out_dir is not False and not os.path.exists(self.out_dir):
                print('Creating subdirectory for behavior')
                os.makedirs(self.out_dir)
    
        # define weighting & decision type options
        self.weight_types = {'':'constant', '_linear-decay':'linear', '_exponential-decay':'exponential'}
        if weight_types is None:    self.wts = ['']
        elif weight_types == 'all': self.wts = list(self.weight_types.keys())
        else:                       self.wts = weight_types
        
        self.decision_types = {'': self.cumulative_coords, '_prev': self.previous_decisions, '_cf': self.counterfactual_decisions}
        if decision_types is None:  self.dts = ['']
        elif weight_types == 'all': self.dts = list(self.decision_types.keys())
        else:                       self.dts = decision_types

    def run(self):
        [self.compute_character(c) for c in [1,2,3,4,5,9]]
        self.compute_across_character()
        if self.out_dir is not False:
            self.behavior.to_excel(Path(f'{self.out_dir}/snt_{self.sub_id}_behavior.xlsx'), index=False)
        return self.behavior

    def weight_decisions(self, decisions, decay='constant'):
        ''' 
            different ways to weight the decisions 
        '''
        nt = len(decisions)
        if decay == 'constant':
            weights = np.ones(nt)[:,None] # standard weighting with a constant
        elif decay == 'linear':
            weights = utils.linear_decay(1, 1/nt, nt)[:,None]
        elif decay == 'exponential':
            weights = utils.exponential_decay(1, 1/nt, nt)[:,None]

        return decisions * weights

    def cumulative_coords(self, decisions):
        ''' mainly just to be able to pass a function like for prev decision etc'''
        return decisions.astype(float), np.cumsum(decisions,0).astype(float)

    def previous_decisions(self, decisions, by=1, replace=0):
        # if on each trial the subj represents the last chosen decision/coordinates

        # shift decisions & get coords
        shifted_decisions      = np.ones_like(decisions) * replace
        shifted_decisions[by:] = np.array(decisions)[0:-by] # replace
        shifted_coords         = np.cumsum(shifted_decisions, axis=0)
        return shifted_decisions.astype(float), shifted_coords.astype(float)

    def counterfactual_decisions(self, decisions):
        # - counterfactual wrt each trial
        # -- eg, if on each trial the subj represents the [ultimately] non-chosen decision/coordinates
        coords       = np.cumsum(decisions, 0)
        prev_coords  = coords - decisions # for each trial, undo the decisions by subtracting to get the prev coords
        decisions_cf = -decisions # counterfactual decision
        coords_cf    = prev_coords + decisions_cf # counterfactual coordinates: add the cf decision to the previous coordinate 

        return decisions_cf.astype(float), coords_cf.astype(float)

    def cumulative_mean(self, values, counts):
        # divide sum at each time point by the response count
        cum_sum   = np.cumsum(values, axis=0)
        cum_count = np.cumsum(counts, axis=0)
        return  [cum_sum / cum_count, (cum_sum, cum_count)]

    def minmax_coords(self, decisions):
        ''' 
            find the max & min coordinates 
            incrementally accounts for non-responses
            incremental mean: accumulated choices / number of choices made, at each time point
        '''
        decisions = np.array(decisions)
        
        # most consistency possible [for decision pattern]
        con = abs(decisions)
        con_coords = np.cumsum(con, axis=0)

        # least consistency possible [for decision pattern]
        incon = np.zeros_like(con)
        for ndim in range(2):
            mask = con[:, ndim] != 0 
            con_d = con[mask, ndim] # if its not 0, its a response
            incon[mask, ndim] = [n if not i % 2 else -n for i,n in enumerate(con_d)] # flip signs
        incon_coords = np.cumsum(incon, axis=0)

        # want the cumulative mean to control for non-responses: divide by num of responses to that point
        resp_counts = np.cumsum(abs(decisions) != 0, axis=0) 

        return [incon_coords / resp_counts, con_coords / resp_counts]

    def cumulative_consistency(self, decisions, resp_mask):

        # 1d consistency = abs value coordinate, scaled by min and max possible coordinate  
        minmax = self.minmax_coords(decisions)  
        cum_mean, _ = self.cumulative_mean(decisions, resp_mask)
        consistency_1d = (np.abs(cum_mean) - minmax[0]) / (minmax[1] - minmax[0])

        # 2d consistency = decision vector length, scaled by min and max possible vector lengths
        minmax_r = (np.array([np.linalg.norm(v) for v in minmax[0]]), np.array([np.linalg.norm(v) for v in minmax[1]]))
        cum_mean_r = np.array([np.linalg.norm(v) for v in cum_mean])
        consistency_r = (cum_mean_r - minmax_r[0]) / (minmax_r[1] - minmax_r[0])

        # return both dimensions separately & 2d
        return [consistency_1d, consistency_r]

    def add_3rd_dimension(self, U, V, ori):
        ''' 
            add 3rd dimension to U & V, as well as to the origin coordinates
            U is vector of interest, z-axis will vary w/ number of interactions
            ori will be subtracted from U, so its z-axis will also vary w/ number of interactions
            V is reference vector, z-axis will remain fixed
        '''
        if V.ndim == 2:   V = V[0]
        if ori.ndim == 1: ori = ori[np.newaxis]

        num = np.arange(1, len(U) + 1)[:,np.newaxis]
        U   = np.concatenate([U, num], axis=1) # changes w/ num of interactions
        V   = np.repeat(np.hstack([V, len(U)])[np.newaxis], len(U), axis=0) # fixed
        ori = np.concatenate([np.repeat(ori, len(U), axis=0), num], axis=1) # changes w/ num of interactions   

        return U, V, ori                           

    def quadrant_overlap(self, coords):
        ''' 
            the percentage overlap of the decision polygon (made by vertices of coordinates) with each of the 4 2D quadrants 
            sum == 100
        '''
        
        make_polygon = lambda C : geometry.Polygon(C[sp.spatial.ConvexHull(C).vertices])

        quadrants = {'1': np.array([[0,0], [0,6],  [6,0],  [6,6]]),   
                     '2': np.array([[0,0], [0,6],  [-6,0], [-6,6]]),
                     '3': np.array([[0,0], [0,-6], [-6,0], [-6,-6]]), 
                     '4': np.array([[0,0], [0,-6], [6,0],  [6,-6]])}

        try: 
            polygon = make_polygon(coords)
            overlap = []
            for q, qC in quadrants.items():
                Q = make_polygon(qC)
                overlap.append(polygon.intersection(Q).area/polygon.area)
        except:
             overlap = [np.nan, np.nan, np.nan, np.nan]
        
        return overlap

    def compute_character(self, c):
        ''' 
            compute w/in character 
        '''
        ixs   = np.where(self.behavior['char_role_num']==c)[0]
        behav = self.behavior.loc[ixs,:]

        # masks for dimensions & for responses
        responded = (behav['button_press'] != 0).values[:,np.newaxis] # dont count non-responses for averages, etc
        dim_mask  = np.vstack([(behav['dimension'] == 'affil').values, (behav['dimension'] == 'power').values]).T # boolean
        resp_mask = np.multiply(dim_mask, responded) # boolean

        # get decisions into shape=(12,2) [or (3,2) for neutral]
        if 'affil' not in behav.columns: # backward compatability
            decisions = behav['decision'].values[:,np.newaxis] * (dim_mask * 1) 
        else:
            decisions = behav[['affil','power']].values 
        if decisions.shape != (12,2) and decisions.shape != (3,2):
            raise Exception(f"Character number {c}'s decision array {decisions.shape} is wrong shape")    
        if decisions.shape != resp_mask.shape:
            raise Exception(f"Character number {c}'s decision {decisions.shape} and response mask {resp_mask.shape} arrays are diff. shapes")

        #-----------------------------------------
        # weight decisions & calculate coordinates
        #-----------------------------------------
        
        for wt in self.wts:

            ### weight decision updates
            decisions_w = self.weight_decisions(decisions, decay=self.weight_types[wt])
            
            for dt in self.dts:
                
                ### compute specific decision types & coordinates
                decisions, coords = self.decision_types[dt](decisions_w)
                if decisions.shape != (12,2) and decisions.shape != (3,2):
                    raise Exception(f"Character number {c}'s {wt}{dt} decision {decisions.shape} array is wrong shape")
                if decisions.shape != coords.shape:
                    raise Exception(f"Character number {c}'s {wt}{dt} decision {decisions.shape} and coordinates {coords.shape} arrays are diff. shapes")

                self.behavior.loc[ixs, [f'affil{wt}{dt}', f'power{wt}{dt}']]             = decisions
                self.behavior.loc[ixs, [f'affil_coord{wt}{dt}', f'power_coord{wt}{dt}']] = coords

                #--------------------------
                # means and absolute values
                #--------------------------

                ### cumulative mean along each dimension [-1,+1] ###
                cum_mean, (cum_sum, cum_count) = self.cumulative_mean(decisions, resp_mask)
                if c != 9 and np.all(cum_mean[-1,:] != cum_sum[-1,:] / cum_count[-1,:]):
                    raise Exception(f'Cumulative mean for character number {c} is off')
                self.behavior.loc[ixs, [f'affil_mean{wt}{dt}', f'power_mean{wt}{dt}']] = cum_mean

                ### culumative consistency: how far did they go in the direction they were traveling? [0,1] ###
                consistency, consistency_2d = self.cumulative_consistency(decisions, resp_mask)
                if consistency.shape != (12,2) and consistency.shape != (3,2):
                    raise Exception(f'Cumulative consistency for character number {c} is off: {consistency.shape}')
                self.behavior.loc[ixs, [f'affil_consistency{wt}{dt}', f'power_consistency{wt}{dt}']] = consistency[:,0][np.newaxis], consistency[:,1][np.newaxis]
                self.behavior.loc[ixs, f'consistency{wt}{dt}'] = consistency_2d

                #--------------------------------------
                # multi-dimensional: angles & distances
                #--------------------------------------

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
                int_num    = np.arange(1, len(behav)+1)[np.newaxis]

                for ndim in [2, 3]:    
                    for ot in ['neu', 'pov']:

                        ## angles between ori-poi & ori-ref
                        poi = coords
                        ref_frame = ref_frames[ot]
                        ref, ori, drn = ref_frame['ref'], ref_frame['ori'], ref_frame['dir']

                        # if 3d, add 3rd dimension
                        if ndim == 3: 
                            poi, ref, ori = self.add_3rd_dimension(poi, ref[0], ori)
                            drn = None # may not be correct for neutral origin... not sure yet

                        # account for origin
                        poi = poi - ori
                        ref = ref - ori

                        # angle between ori-poi & ori-ref
                        self.behavior.loc[ixs, f'{ot}{ndim}d_angle{wt}{dt}'] = utils.calculate_angle(poi, ref, force_pairwise=False, direction=drn) # outputs radians

                        ## distances from ori-poi
                        if ndim == 2: 
                            self.behavior.loc[ixs, f'{ot}_distance{wt}{dt}'] = [np.linalg.norm(v) for v in poi] # l2 norm is euclidean distance from ori

    def compute_across_character(self):
        ''' 
            variables across characters
        '''
        
        for sx in utils.flatten_nested_lists([[f'{wt}{dt}' for dt in self.dts for wt in self.wts]]):
            for t in range(0, 63):

                X = self.behavior.loc[:t, [f'affil_coord{sx}', f'power_coord{sx}', 'char_decision_num']].values

                try: # 3d 
                    vol = sp.spatial.ConvexHull(X).volume
                except: 
                    vol = np.nan
                try: # 2d
                    shape = sp.spatial.ConvexHull(X[:,0:2]) 
                    perim, area = shape.area, shape.volume # outputs perim & area when 2D
                except: 
                    perim, area = np.nan, np.nan

                overlap = self.quadrant_overlap(X[:,0:2])
                if not np.isnan(overlap[0]) and not math.isclose(np.sum(overlap), 1):
                    raise Exception(f"Sum of quadrant overlap percentages isn't 100%: {np.sum(overlap)*100}%")

                self.behavior.loc[t, f'perimeter{sx}']  = perim
                self.behavior.loc[t, f'area{sx}']       = area
                self.behavior.loc[t, f'volume{sx}']     = vol
                self.behavior.loc[t, f'Q1_overlap{sx}'] = overlap[0]
                self.behavior.loc[t, f'Q2_overlap{sx}'] = overlap[1]
                self.behavior.loc[t, f'Q3_overlap{sx}'] = overlap[2]
                self.behavior.loc[t, f'Q4_overlap{sx}'] = overlap[3]
                

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
        assert utils.is_numeric(sub_id), 'Subject id isnt numeric; check that filename has this pattern: "snt_subid*.xlsx"'

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
  