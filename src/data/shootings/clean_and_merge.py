import os
import sys
import datetime as dt

import yaml
import numpy as np
import pandas as pd
from fuzzywuzzy import process

# Load configuration file specifting data paths and filenames
project_directory = 'grassroots_law'
_project_root = os.getcwd()\
    .split(project_directory)[0]\
    + project_directory + '/'
_CONFIG_FILE = _project_root + 'config.yml'
with open(_CONFIG_FILE,'r') as f:
        _cfg = yaml.safe_load(f)

sys.path.append(_project_root+'src/')

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
    p_roject_root = os.getcwd()\
        .split(project_directory)[0]\
        + project_directory + '/'
    
    STATES_FILE = _project_root + 'states.yml'
    with open(STATES_FILE,'r') as f:
            states_dict = yaml.safe_load(f)

    COUNTIES_FILE = _project_root + 'states_counties.yml'
    with open(COUNTIES_FILE,'r') as f:
            counties_dict = yaml.safe_load(f)

    return states_dict, counties_dict


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
    p_roject_root = os.getcwd()\
        .split(project_directory)[0]\
        + project_directory + '/'
    
    STATES_FILE = _project_root + 'states.yml'
    with open(STATES_FILE,'r') as f:
            states_dict = yaml.safe_load(f)

    COUNTIES_FILE = _project_root + 'states_counties.yml'
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
                Name:'links', dtype: object
                Name:'summary' dtype: object
    """
    files = os.listdir('{}/data/raw'.format(_project_root))
    # print(files)
    df = pd.DataFrame()

    usecols = _cfg['usecols']

    for f in files:
        d = pd.read_csv(
                    '{}/data/raw/{}'.format(_project_root, f),
                    header=1,
                    )
        # print(d.columns)
        # print(d.shape)
        df = df.append(
                d,
                ignore_index=True
            )
    # Drop duplicated 'summary' columns
    
    is_summary = list(map(lambda x: x=='summary', df.columns.values))
    # print(df.columns.values)
    # wait = input("PRESS ENTER TO CONTINUE.")
    if np.sum(is_summary) > 1:
        inds = [i for i,x in enumerate(n_summary) if x == True]
        to_drop = inds[1:]
        
        df.drop(labels=to_drop, axis=1, inplace=True)
    to_drop = [n for n in df.columns.values if 'Unnamed' in n]
    df.drop(labels=to_drop, axis=1, inplace=True) 
    
    link_cols = [col for col in df.columns.values if 'link' in col]
    links = df[link_cols]
    links_merged = links.apply(
        lambda x: ' '.join([str(n) for n in x.values if str(n)!='nan']),
        axis=1
        )
    df['links'] = links_merged
    df.drop(labels=link_cols, axis=1, inplace=True)

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
        # print(x) # Uncomment to debug keyerrors etc
        state = x[0]
        county = str(x[1]).strip().lower().capitalize()
        
        if county == 'Nan':
            return 'nan'
        if state.lower() == 'nan':
            return '?' + county

        counties = counties_dict['state'][state]['counties']['county']

        match = process.extractOne(county, counties)
        if match[1] < 75:
            return '?' + county
        else:
            return match[0][:-7]
    
    state_county = [tuple(x) for x in df.loc[:,['state', 'county']].to_numpy()]
    df['county'] = list(map(correct_county, state_county))

    return df


def dates_to_datetime(df):
    """
    Convert type of date values to np.datetime64
    """
    def dt_or_nat(x):
        try:
            x = pd.to_datetime(x)
            return x
        except:
            return np.datetime64('NaT')


def main():
    """
    [summary]
    """
    states_dict, counties_dict = load_states()
    df = load_data()
    df = clean_states(df, states_dict)
    df = clean_counties(df, counties_dict)
    
    # Reindex columns to desired order
    writecols = _cfg['writecols']
    df = df[writecols]

    # Order dataframe by state/date

    # df['date'] = list(map(pd.to_datetime, df['date']))
    df.sort_values(by=['state','date','county'], inplace=True)
    
    # Write clean, merged data to Google Sheets
    sheet_id = _cfg['merged_data']['sheet_id']
    cell_range = _cfg['merged_data']['sheet_name']
    data = [[str(m) for m in n] for n in df.to_numpy()]
    data.insert(0, list(df.columns.values))

    # gs_write(data, sheet_id, cell_range)

    return df


if __name__ == '__main__':
    data = main()

