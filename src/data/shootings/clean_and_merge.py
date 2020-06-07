import os
import sys
import datetime as dt

import yaml
import numpy as np
import pandas as pd
from fuzzywuzzy import process

# Load configuration file specifting data paths and filenames
project_directory = 'grassroots_law'
project_root = os.getcwd()\
    .split(project_directory)[0]\
    + project_directory + '/'
_CONFIG_FILE = project_root + 'config.yml'
with open(_CONFIG_FILE,'r') as f:
        _cfg = yaml.safe_load(f)

sys.path.append(project_root+'src/')

from data.google_sheets import gs_write


def load_states():
    """
    Load dicts mapping states to abbreviations and list of counties.

    Load dicts mapping correctly spelled State names to abbreviated State
    codes (e.g. California, CA; Tennessee, TN etc) and State names to County
    names within each State.

    Returns:
        states_dict (dict): 
            {'AK': {'name': 'Alaska', 'abbreviation': 'AK'}, ...}]
        counties_dict (dict):
            {'state': {'Alabama': {'abv': 'AL', 'counties': {'county': [Autauga County, ...]}, ...}}
    """
    project_directory = 'grassroots_law'
    project_root = os.getcwd()\
        .split(project_directory)[0]\
        + project_directory + '/'
    
    STATES_FILE = project_root + 'states.yml'
    with open(STATES_FILE,'r') as f:
            states_dict = yaml.safe_load(f)

    COUNTIES_FILE = project_root + 'states_counties.yml'
    with open(COUNTIES_FILE,'r') as f:
            counties_dict = yaml.safe_load(f)

    return states_dict, counties_dict


def load_data():
    """
    Import regional police shootings CSVs from local directory

    Returns:
        df (pd.DataFrame): 
            Index:
                RangeIndex
            Columns:
                Name: 'state', dtype: object
                Name:'date', dtype: object
                Name:'victim_name', dtype: object
                Name:'officer_name', dtype: object
                Name:'armed_unarmed', dtype: object
                Name:'cause_of_death', dtype: object
                Name:'officer_charged', dtype: object
                Name:'alleged_crime', dtype: object
                Name:'county', dtype: object
                Name:'link_1', dtype: object
                Name:'summary' dtype: object
    """
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


def clean_states(df, states_dict):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        states_dict ([type]): [description]

    Returns:
        df ([])
    """
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

    df['state'] = list(map(correct_state, df['state'])) 
    df.insert(1, 'state_code', df['state'])

    df['state'] = list(map(return_long, df['state']))
    df['state_code'] = list(map(return_short, df['state_code']))

    return df


def clean_counties(df, counties_dict):
    
    def correct_county(x):
        """[summary]

        Args:
            x (tuple): (state_name (clean), county)
        Returns:
            (str): Correctly spelled and formatted
                                county name.
        """
        print(x)
        state = x[0]
        county = str(x[1]).strip().lower().capitalize()
        
        if county == 'Nan':
            return 'nan'
        if state.lower() == 'nan':
            return '?' + county

        counties = counties_dict['state'][state]['counties']['county']
        match = process.extractOne(county, counties)
        if match[1] > 0.75:
            return '?' + county
        else:
            return match[0]
    
    state_county = [tuple(x) for x in df.loc[:,['state', 'county']].to_numpy()]
    df['county'] = list(map(correct_county, state_county))

    return df


def main():
    """
    [summary]
    """
    states_dict, counties_dict = load_states()
    df = load_data()
    df = clean_states(df, states_dict)
    df = clean_counties(df, counties_dict)
    
    sheet_id = _cfg['merged_data']['sheet_id']
    cell_range = _cfg['merged_data']['sheet_name']
    data = [[str(m) for m in n] for n in df.to_numpy()]
    data.insert(0, list(df.columns.values))

    gs_write(data, sheet_id, cell_range)


if __name__ == '__main__':
    data = main()

