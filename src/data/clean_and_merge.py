import os
import sys
import datetime as dt
import yaml

import numpy as np
import pandas as pd
from fuzzywuzzy import process


def load_states():
# Load configuration file specifting data paths and filenames
    project_directory = 'grassroots_law'
    project_root = os.getcwd()\
        .split(project_directory)[0]\
        + project_directory + '/'
    _STATES_FILE = project_root + 'states.yml'
    with open(_STATES_FILE,'r') as f:
            states_dict = yaml.safe_load(f)
    return states_dict


def load_data():

    files = os.listdir('../../data/raw')
    # print(files)
    df = pd.DataFrame()

    usecols = [
        'state',
        'date',
        'victim_name',
        'officer_name',
        'armed_unarmed',
        'cause_of_death',
        'officer_charged',
        'alleged_crime',
        'county',
        'link_1',
        'summary'
    ]

    for f in files:
        d = pd.read_csv(
                    '../../data/raw/{}'.format(f),
                    header=1,
                    usecols=usecols
                    )
        # print(d.columns)
        # print(d.shape)
        df = df.append(
                d,
                ignore_index=True
            )
    return df

# for col in usecols:
#     if col == 'date':
#         # dfcol] = pd.to_datetime(df[col])
#         pass
#     else:
#        df[col] = df[col].astype(str)

def clean_states(df, states_dict):
    
    def return_long(x):
        for state in long_short:
            if x in state:
                return state[0]
        return 'nan'

    def return_short(x):
        for state in long_short:
            if x in state:
                return state[1]
        return 'nan'

    def correct_state(x):
        x=str(x).capitalize()
        
        if x.lower() == 'nan':
            return 'nan'
        
        if x == '':
            return x

        if len(x) == 2:
            if x in states_short:
                return x.strip()
            else:
                return process.extractOne(x.strip(), states_short)[0]
        else:
            if x in states_long:
                return x
            else:
                return process.extractOne(x.strip(), states_short)[0]


    states_long = [
        states_dict[key]['name'] for key in states_dict.keys()
        ]

    states_short = [
        states_dict[key]['abbreviation'] for key in states_dict.keys()
        ]

    long_short = [
        [
            states_dict[key]['name'], 
            states_dict[key]['abbreviation']
            ] \
        for key in states_dict.keys()
    ]
    print(long_short)
    df['state'] = list(map(correct_state, df['state'])) 
    df.insert(1, 'state_code', df['state'])

    df['state'] = list(map(return_long, df['state']))
    df['state_code'] = list(map(return_short, df['state_code']))

    return df

def main():
    """
    [summary]
    """
    states_dict = load_states()
    df = load_data()
    df = clean_states(df, states_dict)

    df.to_csv('../../data/interim/killings_data_merged.csv')

if __name__ == '__main__':
    main()