#import_data.py
# Matthew J Cronin, 2020
# matthew.j.cronin@gmail.com
#
"""
Pull data from the Grassroots Law Project Police Shooting tracker

Returns:
    Regional CSV files saved to disk.
"""

import os
import sys
import logging
logging.getLogger(
    'googleapicliet.discovery_cache'
    ).setLevel(logging.ERROR)
    
import streamlit as st
import yaml
import numpy as np
import pandas as pd

# Load configuration file specifting data paths and filenames
project_directory = 'grassroots_law'
project_root = os.getcwd()\
    .split(project_directory)[0]\
    + project_directory + '/'
_CONFIG_FILE = project_root + 'config.yml'
with open(_CONFIG_FILE,'r') as f:
        _cfg = yaml.safe_load(f)

sys.path.append(project_root+'src/')

from data.google_sheets import gs_read

@st.cache
def load_data():
    """
    Pull data from the regional police shootings Google Sheets

    Returns:
        region_data [dict]: dictionary containing arrays of data pulled from
        the Google Sheets API. Keys are region names as described in config.yml
    """

    sheets = _cfg['google_sheets']
    region_data = {}
    for key in sheets.keys():

        sheet = sheets[key]['sheet_id']
        cell_range = sheets[key]['sheet_name']
        cols_row = sheets[key]['cols_row']
        data_start = sheets[key]['data_start']

        dat = gs_read(sheet, cell_range)
        cols =  dat[cols_row]
        n = 1
        for i,col in enumerate(cols):
            if col.lower().strip() == 'link':
                cols[i] = cols[i].strip() + '_'+str(n)
                n+=1
        mappings = sheets[key]['col_mappings']
        print(key)
        print(cols)
        cols = [mappings[n] for n in cols]

        data = dat[data_start:]
        data.insert(0,cols)

        region_data[key] = data

    return region_data


def main():
    
    region_data = load_data()

    for k in region_data.keys():
        d = pd.DataFrame(region_data[k])
        d.to_csv('../../data/raw/{}-2020-06-05.csv'.format(k))


if __name__ == '__main__':
    main()

