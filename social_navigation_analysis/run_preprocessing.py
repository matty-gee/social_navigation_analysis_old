# TODO: run through all samples with this as a .py...
# more unittests...

def main():

    import info as snt_info
    import preprocess as snt_preprc
    import pandas as pd
    import datetime, glob, os

    synapse_dir = '/Volumes/synapse/projects/SocialSpace/Projects/'
    datasets = pd.read_excel(f'{synapse_dir}/SNT_datasets.xlsx')

    #---------------------------
    # select the project details
    #---------------------------

    # for proj_ix in range(length(datasets)):
    proj_ix = 1
    project_dir = datasets.loc[proj_ix,'base_directory']
    file_format = datasets.loc[proj_ix,'raw_file_format']

    # directories etc
    datasets.loc[proj_ix, 'raw_directory']        = f'{file_format}'
    datasets.loc[proj_ix, 'organized_directory']  = f'Organized'
    datasets.loc[proj_ix, 'behavioral_directory'] = f'Behavior'
    datasets.loc[proj_ix, 'timing_directory']     = f'Timing'
    datasets.loc[proj_ix, 'last_processed']       = datetime.datetime.now()

    # find files
    raw_files = [f for f in glob.glob(f"{project_dir}/{datasets.loc[0, 'raw_directory']}/*") if '~$' not in f]
    n_raw     = len(raw_files)
    print(f'Found {n_raw} files')
    for raw_fname in raw_files:
        
        sub_id = raw_fname.split('/')[-1].split('.')[0].split('_')[1]

        #-----------------------
        # 1 - parse the raw file 
        #-----------------------
        
        xlsx_fname = f"{project_dir}/{datasets.loc[proj_ix, 'organized_directory']}/SNT_{sub_id}.xlsx"
        if not os.path.exists(xlsx_fname):
            
            print(f'{sub_id}: parsing', end='\r')
            if file_format == 'Logs':
                snt_preprc.parse_log(raw_fname, experimenter=datasets.loc[0,'experimenter'], output_timing=True, out_dir=project_dir)
            elif file_format == 'CSVs':
                snt_preprc.parse_csv(raw_fname, snt_version=datasets.loc[0,'options_version'], out_dir=project_dir)

        #---------------------
        # 2 - compute behavior
        #---------------------
        
        behav_fname = f"{project_dir}/{datasets.loc[proj_ix, 'behavioral_directory']}/SNT_{sub_id}_behavior.xlsx"
        if not os.path.exists(behav_fname):
            
            print(f'{sub_id}: computing behavior', end='\r')
            snt_preprc.compute_behavior(xlsx_fname, weight_types=True, decision_types=True, coord_types=True, out_dir=project_dir)

    # count number of files in each
    n_xlsx      = len([f for f in glob.glob(f"{project_dir}/{datasets.loc[proj_ix, 'organized_directory']}/*") if '~$' not in f])
    n_timing    = len([f for f in glob.glob(f"{project_dir}/{datasets.loc[proj_ix, 'timing_directory']}/*") if '~$' not in f])
    behav_files = [f for f in glob.glob(f"{project_dir}/{datasets.loc[proj_ix, 'behavioral_directory']}/*") if '~$' not in f]
    n_behavior  = len(behav_files)

    if n_raw != n_xlsx: 
        print('There are missing subjects')

    datasets.loc[proj_ix, 'n_raw_files']       = n_raw
    datasets.loc[proj_ix, 'n_organized_files'] = n_xlsx
    datasets.loc[proj_ix, 'n_timing_files']    = n_timing
    datasets.loc[proj_ix, 'n_behavior_files']  = n_behavior

    #------------------------------
    # 3 - summarize across subjects
    #------------------------------

    snt_preprc.summarize_behavior(behav_files, out_dir=project_dir)

    datasets.to_excel(f'{synapse_dir}/SNT_datasets.xlsx', index=False)

    # TODO: auto-combine w. self-reports, dots etc...

if __name__ == "__main__":
    main()