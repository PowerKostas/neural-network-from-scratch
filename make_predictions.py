import os
import pickle
import feed_forward

def get(prediction_input_values):
    # Retrieve saved model info
    with open(os.path.dirname(os.path.abspath(__file__))+'/digit_recognition/model.pkl', 'rb') as f:
        model = pickle.load(f)

    hidden_nodes = model['hidden_nodes']
    activation_type = model['activation_type']
    weights = model['weights']
    biases = model['biases']
    problem_type = model['problem_type']


    # Make predictions
    _, _, _, _, predicted_values = feed_forward.start(prediction_input_values, 10, hidden_nodes, activation_type, weights, biases, problem_type)
    return predicted_values
