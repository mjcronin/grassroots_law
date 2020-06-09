import pandas as pd


# DATA SOURCE https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv
data = None
with open("us_cities_states_counties.csv", "r") as datafile:
    data = pd.read_csv(datafile, delimiter='|')

data.columns = ['_'.join(col.lower().split(' ')) for col in data.columns]
data.to_csv("us-locations.csv", index=False)


# COLUMN NAMES: City,State short,State full,County,City alias
# find unique set for each attribute

unique_col = lambda colname : (colname, set(data[colname]))

unique = [ unique_col(colname) for colname in data.columns ]

#write all unique data to its own files
for colname, values in unique:
    tmp_df = pd.DataFrame(values) #create dataframe so the data is indexed
    tmp_df.columns = [colname]
    tmp_df = tmp_df[pd.notnull(tmp_df[colname])]
    tmp_df = tmp_df.apply(lambda r : r[colname].lower(), axis=1)
    tmp_df.to_csv("./dictionaries/{}.txt".format(colname), index=True)


#create vectorized combinations for all real location info
#real_combinations = 