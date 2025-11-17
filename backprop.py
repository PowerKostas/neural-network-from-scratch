import numpy as np
import cost_functions
import activation_functions

def start(weights, biases, true_values, activated_nn_values, nn_values, cost_type, activation_type, lr, problem_type):
    # For the math: https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc
    # Calculates dC_dZ for the output layer only, it uses a shortcut for CCE with softmax
    if cost_type != 'CCE':
        dC_dA = cost_functions.func(true_values, activated_nn_values[-1], cost_type, 1)
        dA_dZ = activation_functions.func(nn_values[-1], activated_nn_values[-1], activation_type, 1)
        dC_dZ = dC_dA * dA_dZ

    else:
        dC_dZ = activated_nn_values[-1] - true_values

    
    # Backpropagates from the output layer to the first hidden layer, the new dC_dZ is the propagator that helps us get the weight and bias
    # gradients for each new layer
    for i in range(len(activated_nn_values) - 1, 0, -1):
        dC_dW = dC_dZ @ activated_nn_values[i - 1].T
        dC_db = np.sum(dC_dZ, axis = 1, keepdims = True)
        dC_dZ = (weights[i - 1].T @ dC_dZ) * activation_functions.func(nn_values[i - 1], activated_nn_values[i - 1], activation_type, 1)

        # Gradient clipping
        if problem_type == 'Regression': 
            dC_dW = np.clip(dC_dW, -true_values.mean(), true_values.mean())
            dC_db = np.clip(dC_db, -true_values.mean(), true_values.mean())


        # Gradient descent for the weights and biases (go in the opposite direction of the gradient, eg: if it's a positive gradient go negative, 
        # if it's a negative gradient go positive)
        weights[i - 1] += -lr * dC_dW
        biases[i - 1] += -lr * dC_db

    return weights, biases
        