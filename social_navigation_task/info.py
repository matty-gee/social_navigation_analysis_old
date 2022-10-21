import pandas as pd 
from pathlib import Path 
import openpyxl

pkg_dir      = str(Path(__file__).parent.absolute())
example_log  = str(Path(f'{pkg_dir}/../data/example_subject/snt_18001.log'))
example_xlsx = str(Path(f'{pkg_dir}/../data/example_subject/snt_18001.xlsx'))

# decision info:
try: 

    # standard details file
    task_file = str(Path(f'{pkg_dir}/../data/snt_details.xlsx'))
    task = pd.read_excel(task_file)
    task.sort_values(by='slide_num', inplace=True)
    decision_trials = task[task['trial_type'] == 'Decision']
    convert_dict = {'decision_num': int,
                    'scene_num': int,
                    'char_role_num': int,
                    'char_decision_num': int,
                    'cogent_onset': float}
    decision_trials = decision_trials.astype(convert_dict)
    decision_trials.reset_index(inplace=True, drop=True)

    # validated decisions, w/ alphabetically sorted options - for parsing online data w/ randomized options
    standard = pd.read_excel(f'../data/snt_sorted-options_standard_mem50_n81.xlsx')
    standard = standard.sort_values(by = 'decision_num').reset_index(drop=True)
    schema = pd.read_excel(f'../data/snt_sorted-options_schema.xlsx')
    schema = schema.sort_values(by = 'decision_num').reset_index(drop=True)
    adolescent = pd.read_excel(f'../data/snt_sorted-options_adolescent.xlsx')
    adolescent = adolescent.sort_values(by = 'decision_num').reset_index(drop=True)
    validated_decisions = {'standard': standard, 'schema': schema, 'adolescent': adolescent}

except: 
    raise Exception(f"Can't find: '{pkg_dir}/../data/snt_details.xlsx'")

# defaults
character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss', 'neutral'] # in order of role num in snt_details