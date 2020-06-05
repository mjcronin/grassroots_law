import os
import sys

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
    sheets = _cfg['google_sheets']
    test=3
    sheet = sheets['northeast']['sheet_id']
    cell_range = sheets['northeast']['sheet_name']
    test = gs_read(sheet, cell_range)
    test = [x[:12] for x in test]

    cols = test[sheets['northeast']['cols_row']]
    print(cols)
    n=1
    for i,col in enumerate(cols):
        if col.lower().strip() == 'link':
            cols[i] = cols[i].strip() + '_'+str(n)
            n+=1
    data = test[sheets['northeast']['data_start']:]
    
    return pd.DataFrame(data=data, columns=cols)

# @st.cache
def check_cols():
    sheets = _cfg['google_sheets']

    # print(sheets)
    cols_check = []
    test=2 
    for key in sheets.keys():
        # print(key)
        sheet = sheets[key]['sheet_id']
        # print(sheet)
        cell_range = sheets[key]['sheet_name']
        cols_row = sheets[key]['cols_row']
        dat = gs_read(sheet, cell_range)
        cols =  dat[cols_row]
        n = 1
        for i,col in enumerate(cols):
            if col.lower().strip() == 'link':
                cols[i] = cols[i].strip() + '_'+str(n)
                n+=1
        cols_check.append(cols)

    # print(cols_check)

    return(cols_check)

def main():
    st.title('Grassroots Law Project Data Exploration')
    df_test = load_data()

    # st.write(df_test.head())
    # st.write(df_test.describe())
    cols_check = check_cols()
    
    print(all([n == cols_check[0] for n in cols_check]))
    
    st.write(pd.DataFrame(data=cols_check))


if __name__ == '__main__':
    main()

