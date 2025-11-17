import numpy as np

# Derivative = 0 = Cost | Derivative = 1 = The partial derivative of the cost with respect to the last layer of activated nodes (dC / dA^L)
def func(true_values, predicted_values, cost_type, derivative):
    if cost_type == 'MAE':
        if derivative == 0:
            return np.mean(np.abs(true_values - predicted_values))
        
        else:
            return np.sign(predicted_values - true_values) / predicted_values.shape[1]
    
    elif cost_type == 'MSE':
        if derivative == 0:
            return np.mean((true_values - predicted_values) ** 2)
        
        else:
            return (2 * (predicted_values - true_values)) / predicted_values.shape[1]
    
    elif cost_type == 'BCE':
        predicted_values = np.clip(predicted_values, 1e-15, 1 - 1e-15) # Clips predicted values to avoid errors
        if derivative == 0:
            return np.mean(- (true_values * np.log(predicted_values) + (1 - true_values) * np.log(1 - predicted_values)))
        
        else:
            return -(true_values / predicted_values - (1 - true_values) / (1 - predicted_values)) / predicted_values.shape[1]
    
    elif cost_type == 'CCE':
        predicted_values = np.clip(predicted_values, 1e-15, 1 - 1e-15)  # Clips predicted values to avoid errors
        return (-np.sum(true_values * np.log(predicted_values)) / true_values.shape[1])
    