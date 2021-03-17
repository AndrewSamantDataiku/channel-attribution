

### Load Environment

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *
from pychattr.channel_attribution import MarkovModel, HeuristicModel


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
positive_conversion_id = str(variables.get("conversion_event_id",None) )
# Column indicating the order of events
order_feature = variables.get("order_feature",None)
# Column indicating if the path ended in non-conversion (Can be none)
null_feature='null_' + conversion_feature
# Column indicating Revenue Amount (Can be none)
revenue_feature= None
# Column indicating Cost Amount (Can be none)
cost_feature= None


### Set Parameters for Markov Chains

# The memory of the Markov Chain (How many events are the outcomes in 'channel to' outcome)
k_order=variables.get("k_order",None)
# Number of Markov Chain Simulations
n_simulations=variables.get("n_simulations",None)
# Maximum number of steps for a simulated path
max_steps=variables.get("max_steps",None)
# Random Seed for random number generator
random_state=variables.get("random_state",None)



### Set Parameters for Output data

first_touch=True
last_touch=True
linear_touch=True
ensemble_results=True
return_transition_probs=True




### Group data by group column

# Function to concatenate events into lists
sep = '>'
def concat(x):
    return sep.join(x.astype(str).tolist())
# Function to return if a group has a conversion
def get_max(x):
    return positive_conversion_id in x.astype(str).tolist()



df_grouped = df.sort_values(by=order_feature).groupby(group_feature).agg({event_feature:concat, conversion_feature:get_max,null_feature:get_null})
df_grouped = df_grouped.reset_index(drop=True)
df_grouped['null_feature'] = True
df_grouped['null_feature'][df_grouped[conversion_feature]] = False


# Create Markov Model

mm = MarkovModel(path_feature=event_feature,
                 conversion_feature=conversion_feature,
                 null_feature='null_feature',
                 revenue_feature=revenue_feature,
                 cost_feature=cost_feature,
                 separator=sep,
                 k_order=k_order,
                 n_simulations=n_simulations,
                 max_steps=max_steps,
                 return_transition_probs=return_transition_probs,
                 random_state=random_state)
mm.fit(df_grouped)



# Create Heuristic Model

hm = HeuristicModel(path_feature=event_feature,
                    conversion_feature=conversion_feature,
                    null_feature=null_feature,
                    revenue_feature=revenue_feature,
                    cost_feature=cost_feature,
                    separator=sep,
                    first_touch=first_touch,
                    last_touch=last_touch,
                    linear_touch=linear_touch,
                    ensemble_results=ensemble_results)
hm.fit(df_grouped)


# Combine results

output_df = hm.attribution_model_.set_index('channel').join(mm.attribution_model_.set_index('channel_name'))
output_df = output_df.join(mm.removal_effects_.set_index('channel_name')).reset_index()
output_df.rename(columns={ 'channel': event_feature }, inplace = True)


# Save results

transition_matrix = dataiku.Dataset(get_output_names_for_role('transition_matrix')[0])
transition_matrix.write_with_schema(mm.transition_matrix_)
removal_effects = dataiku.Dataset(get_output_names_for_role('channel_attributes')[0])
removal_effects.write_with_schema(output_df)