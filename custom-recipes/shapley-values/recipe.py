### Load Environment

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from itertools import chain, combinations
from collections import defaultdict
from math import factorial
from dataiku.customrecipe import *


### Read input data

dataset = dataiku.Dataset(get_input_names_for_role('input_dataset')[0])
df = dataset.get_dataframe()


### Set Parameters for Input data

# Get user inputs
variables = get_recipe_config()
# Feature with the paths
event_feature=variables.get("event_feature",None)
# Seperator for each item in the path
group_feature = variables.get("group_feature",None)
# Column indicating if the path ended with conversion
conversion_feature=variables.get("conversion_feature",None)
positive_conversion_id = str(variables.get("conversion_event_id",None))
# Column indicating if the path ended in non-conversion (Can be none)
null_feature=None
# Column indicating Revenue Amount (Can be none)
revenue_feature= None
# Column indicating Cost Amount (Can be none)
cost_feature= None

### Group data by group column

# Function to concatenate events into lists
sep = '>'
# Function to return if a group has a conversion
def get_max(x):
    return positive_conversion_id in x.astype(str).tolist()

# Gets unique elements of a list and concatenates them
def unique(l):
    x = np.array(l.astype(str))
    x = np.sort(x)
    return sep.join(np.unique(x))

def powerset(s):
    l = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    return [sep.join(a) for a in l]

# Group by Coalition
print("GROUPING BY USER")
df[event_feature] = df[event_feature].astype(str)
df_grouped_by_user = df.groupby(group_feature).agg({event_feature:unique, conversion_feature:get_max}).reset_index(drop=True)
print("GROUPING BY COALITION")
df_grouped_by_coalition = df_grouped_by_user.groupby(event_feature).agg({conversion_feature:'sum'}).reset_index()

# Get total possible conversions attributable to each coalition
v_values = {}
C_values = dict(zip(df_grouped_by_coalition[event_feature].values,df_grouped_by_coalition[conversion_feature].values))

print("CREATING V VALUES")
for A in df_grouped_by_coalition[event_feature]:
    subsets_of_A = powerset(A.split(sep))
    worth_of_A=0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    v_values[A] = worth_of_A

# Get Shapley Values
channels = df[event_feature].unique()
n=len(channels)
shapley_values = defaultdict(int)
print("GENERATING SHAPLEY VALUES")
for channel in channels:
    for A in v_values.keys():
        if channel not in A.split(sep):
            cardinal_A=len(A.split(sep))
            A_with_channel = A.split(sep)
            A_with_channel.append(channel)
            A_with_channel=sep.join(sorted(A_with_channel))
            if (A_with_channel in v_values) and (A in v_values):
                shapley_values[channel] += (v_values[A_with_channel]-v_values[A])*(factorial(cardinal_A)*factorial(n-cardinal_A-1)/factorial(n))
            # Add the term corresponding to the empty set
            if channel in v_values:
                shapley_values[channel]+= v_values[channel]/n
shapley_df = pd.DataFrame(list(shapley_values.items()),columns=[event_feature,'shapley_value']).fillna(0)



# Save results

attribution_matrix = dataiku.Dataset(get_output_names_for_role('shapley_values')[0])
attribution_matrix.write_with_schema(shapley_df)