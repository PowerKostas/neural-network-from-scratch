import pandas as pd
import numpy as np
import os

# Doing this to import module from parent folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_validation

def get_input_true_values(k):
    # Reads the csv
    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/StressLevelDataset.csv', encoding_errors = 'ignore').sample(frac = 1)


    # For the input, drop the output column and any unwanted columns, for the true values, just get the output column
    input_values = df.drop(columns = ['anxiety_level', 'depression']).copy()
    true_values = df['depression'].copy()


    # Does one hot encoding for any input or true values string column
    #input_values = pd.get_dummies(input_values, columns = ['Model', 'Region', 'Fuel_Type', 'Transmission'])
    #true_values = pd.get_dummies(true_values, columns = ['Species'])


    # Formats input and true values columns
    #input_values['CC/Battery Capacity'] = input_values['CC/Battery Capacity'].str.replace(',','').str.extract(r'(\d+\.?\d*)')
    #input_values['HorsePower'] = input_values['HorsePower'].str.extract(r'(\d+\.?\d*)')
    #input_values['Total Speed'] = input_values['Total Speed'].str.extract(r'(\d+\.?\d*)')
    #input_values['Performance(0 - 100 )KM/H'] = input_values['Performance(0 - 100 )KM/H'].str.extract(r'(\d+\.?\d*)')
    #input_values['Torque'] = input_values['Torque'].str.extract(r'(\d+\.?\d*)')
    #true_values = true_values.map({'B': 0, 'M': 1})


    # Turns both dataframes to float, if any row causes problems, it removes it from both dataframes, if not done already
    input_values = input_values.apply(pd.to_numeric, errors = 'coerce')
    input_values = input_values.dropna()
    true_values = true_values.loc[input_values.index]

    true_values = true_values.apply(pd.to_numeric, errors = 'coerce')
    true_values = true_values.dropna()
    input_values = input_values.loc[true_values.index]

    input_values = input_values.astype(float)
    true_values = true_values.astype(float)


    # Normalizes any input number column
    input_values['self_esteem'] = (input_values['self_esteem'] - input_values['self_esteem'].mean()) / input_values['self_esteem'].std()
    input_values['mental_health_history'] = (input_values['mental_health_history'] - input_values['mental_health_history'].mean()) / input_values['mental_health_history'].std()
    input_values['headache'] = (input_values['headache'] - input_values['headache'].mean()) / input_values['headache'].std()
    input_values['blood_pressure'] = (input_values['blood_pressure'] - input_values['blood_pressure'].mean()) / input_values['blood_pressure'].std()
    input_values['sleep_quality'] = (input_values['sleep_quality'] - input_values['sleep_quality'].mean()) / input_values['sleep_quality'].std()
    input_values['breathing_problem'] = (input_values['breathing_problem'] - input_values['breathing_problem'].mean()) / input_values['breathing_problem'].std()
    input_values['noise_level'] = (input_values['noise_level'] - input_values['noise_level'].mean()) / input_values['noise_level'].std()
    input_values['living_conditions'] = (input_values['living_conditions'] - input_values['living_conditions'].mean()) / input_values['living_conditions'].std()
    input_values['safety'] = (input_values['safety'] - input_values['safety'].mean()) / input_values['safety'].std()
    input_values['basic_needs'] = (input_values['basic_needs'] - input_values['basic_needs'].mean()) / input_values['basic_needs'].std()
    input_values['academic_performance'] = (input_values['academic_performance'] - input_values['academic_performance'].mean()) / input_values['academic_performance'].std()
    input_values['study_load'] = (input_values['study_load'] - input_values['study_load'].mean()) / input_values['study_load'].std()
    input_values['teacher_student_relationship'] = (input_values['teacher_student_relationship'] - input_values['teacher_student_relationship'].mean()) / input_values['teacher_student_relationship'].std()
    input_values['future_career_concerns'] = (input_values['future_career_concerns'] - input_values['future_career_concerns'].mean()) / input_values['future_career_concerns'].std()
    input_values['social_support'] = (input_values['social_support'] - input_values['social_support'].mean()) / input_values['social_support'].std()
    input_values['peer_pressure'] = (input_values['peer_pressure'] - input_values['peer_pressure'].mean()) / input_values['peer_pressure'].std()
    input_values['extracurricular_activities'] = (input_values['extracurricular_activities'] - input_values['extracurricular_activities'].mean()) / input_values['extracurricular_activities'].std()
    input_values['bullying'] = (input_values['bullying'] - input_values['bullying'].mean()) / input_values['bullying'].std()
    input_values['stress_level'] = (input_values['stress_level'] - input_values['stress_level'].mean()) / input_values['stress_level'].std()


    # Turns both dataframes to numpy arrays and transposes them for later
    input_values = input_values.to_numpy().T
    if true_values.ndim == 1:
        true_values = true_values.to_numpy().reshape(1, -1)
    
    else:
        true_values = true_values.to_numpy().T  


    # Cross validation
    train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices = cross_validation.start(input_values, true_values, k)
    return train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices
