"""
Interactive dashboard for visualization of the Grassroots Law Project 
crowdsourced data on police shootings in the United States
"""

import streamlit as st
import pandas as pd
import numpy as np

PAGES = {
    'Map'
}

@st.cache
def load_data():
    data = pd.read_csv(
        'police_shootings_merged_2020 - shootings_data_merged.csv'
        )


def main():
    """Main function of the App"""
    st.title('Grassroots Law Project')
    data = load_data()


if __name__ == "__main__":
    main()