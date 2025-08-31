import numpy as np
import csv
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read Cleveland Heart Disease data
heartDisease = pd.read_csv(r'D:\ML Lab Excersise\exp8\heart.csv')
heartDisease = heartDisease.replace('?', np.nan)

# Display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())

# Model Bayesian Network
Model = DiscreteBayesianNetwork([
    ('age', 'trestbps'),
    ('age', 'fbs'),
    ('sex', 'trestbps'),
    ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('fbs', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
Model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(Model)

# Computing the Probability of HeartDisease given Age
print('\n1. Probability of HeartDisease given Age=29')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 29})
print(q.values) 

# Computing the Probability of HeartDisease given cholesterol
print('\n2. Probability of HeartDisease given cholesterol=126')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 126})
print(q.values)
