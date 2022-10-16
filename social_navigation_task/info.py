import pandas as pd 
from pathlib import Path 
import openpyxl

pkg_dir      = str(Path(__file__).parent.absolute())
example_log  = str(Path(f'{pkg_dir}/../data/example_subject/snt_18001.log'))
example_xlsx = str(Path(f'{pkg_dir}/../data/example_subject/snt_18001.xlsx'))

# decision info:
try: 
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
except: 
    raise Exception(f"Can't find: '{pkg_dir}/../data/find snt_details.xlsx'")

# defaults
character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss'] # in order of role num in decision_detials