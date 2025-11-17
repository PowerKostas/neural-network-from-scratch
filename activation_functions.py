import numpy as np

# Derivative = 0 = Cost | Derivative = 1 = The partial derivative of the last layer of activated nodes with respect to the last layer of 
# non activated nodes (dA^L / dZ^L)
def func(nn_values, activated_nn_values, activation_type, derivative):
    if activation_type == 'Sigmoid':
        if derivative == 0:
            return 1 / (1 + np.exp(-nn_values)) 
        
        else:
            return activated_nn_values * (1 - activated_nn_values)
    
    elif activation_type == 'Tanh':
        if derivative == 0:
            return np.tanh(nn_values)
        
        else:
            return 1 - np.tanh(activated_nn_values) ** 2
    
    elif activation_type == 'ReLU':
        if derivative == 0:
            return np.maximum(0, nn_values)
        
        else:
            return (nn_values > 0).astype(float)
    
    elif activation_type == 'Softmax':
        return np.exp(activated_nn_values) / np.sum(np.exp(activated_nn_values), axis = 0)
