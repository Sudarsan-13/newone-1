import pandas as pd
import numpy as np

# Defining the dataset as a dictionary
data_dict = {
    'Buys_Iphone': ['yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
    'CR': ['fair', 'excellent', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'fair', 'excellent', 'fair', 'excellent', 'excellent'],
    'Age': ['less than 30', 'less than 30', '30 to 40', 'less than 30', '30 to 40', '30 to 40', 'less than 30', 'less than 30', '30 to 40', '30 to 40', '30 to 40', 'less than 30', 'less than 30', '30 to 40'],
    'Income': ['medium', 'high', 'high', 'medium', 'medium', 'medium', 'low', 'low', 'medium', 'low', 'high', 'medium', 'low', 'medium'],
    'Student': ['yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes']
}

# Convert the dictionary to a pandas DataFrame
data = pd.DataFrame(data_dict)

# Print the dataset
print(data)

# Print the column names
print(data.columns)

# Prior Probability: P(Buys_Iphone)
prior = data.groupby('Buys_Iphone').size().div(len(data))
print("Prior Probabilities:")
print(prior)

# Likelihood for each feature given Buys_Iphone = yes or no
likelihood = {}
likelihood['CR'] = data.groupby(['Buys_Iphone', 'CR']).size().div(len(data)).div(prior)
likelihood['Age'] = data.groupby(['Buys_Iphone', 'Age']).size().div(len(data)).div(prior)
likelihood['Income'] = data.groupby(['Buys_Iphone', 'Income']).size().div(len(data)).div(prior)
likelihood['Student'] = data.groupby(['Buys_Iphone', 'Student']).size().div(len(data)).div(prior)

print("\nLikelihoods:")
print(likelihood)

# Probability that the person will buy
p_yes = (likelihood['Age']['yes']['less than 30'] * 
         likelihood['Income']['yes']['medium'] * 
         likelihood['Student']['yes']['yes'] * 
         likelihood['CR']['yes']['fair'] * 
         prior['yes'])

# Probability that the person will NOT buy
p_no = (likelihood['Age']['no']['less than 30'] * 
        likelihood['Income']['no']['medium'] * 
        likelihood['Student']['no']['yes'] * 
        likelihood['CR']['no']['fair'] * 
        prior['no'])

print('\nProbability of Yes:', p_yes)
print('Probability of No:', p_no)

# Normalization
v_yes = p_yes / (p_yes + p_no)
print('After Normalization (Yes):', v_yes)

v_no = p_no / (p_yes + p_no)
print('After Normalization (No):', v_no)
