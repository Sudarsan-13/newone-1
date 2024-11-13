import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Step 1: Define the structure of the Bayesian Network
model = BayesianNetwork([('Rain', 'Wet_Grass'), ('Sprinkler', 'Wet_Grass')])

# Step 2: Prepare the data
data = pd.DataFrame(data={'Rain': ['Yes', 'No', 'Yes', 'No'],
                          'Sprinkler': ['No', 'Yes', 'Yes', 'No'],
                          'Wet_Grass': ['Yes', 'Yes', 'Yes', 'No']})

# Step 3: Define the CPDs using MaximumLikelihoodEstimator
cpd_rain = MaximumLikelihoodEstimator(model, data).estimate_cpd('Rain')
cpd_sprinkler = MaximumLikelihoodEstimator(model, data).estimate_cpd('Sprinkler')
cpd_wet_grass = MaximumLikelihoodEstimator(model, data).estimate_cpd('Wet_Grass')

# Print CPDs
print(cpd_rain)
print(cpd_sprinkler)
print(cpd_wet_grass)

# Step 4: Add the CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)

# Step 5: Validate the model to check if it's correct
model.check_model()

# Step 6: Perform Inference (Instantiate the variable elimination method for inference)
inference = VariableElimination(model)

# Step 7: Perform inference (CASE 1)
result = inference.query(variables=['Rain'], evidence={'Wet_Grass': 'Yes'})
print(result)

# Step 8: Perform inference (CASE 2)
result = inference.query(variables=['Rain'], evidence={'Wet_Grass': 'No'})
print(result)

# Step 9: Perform inference (CASE 3)
result = inference.query(variables=['Sprinkler'], evidence={'Wet_Grass': 'Yes'})
print(result)

# Step 10: Perform inference (CASE 4)
result = inference.query(variables=['Sprinkler'], evidence={'Wet_Grass': 'No'})
print(result)
