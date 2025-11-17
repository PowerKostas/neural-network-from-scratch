import numpy as np
import activation_functions

def start(input_values, output_nodes, hidden_nodes, activation_type, weights, biases, problem_type):
    if len(weights) == 0: # Only does the below if it's the first epoch, otherwise it uses the already made weights and biases array
        # Creates a weights and biases array, per layer, each node in the layer has a 0 bias, each layer has a (layer n * layer n - 1) shape weights
        # array, each x column represents the weights coming out of the layer n - 1 x node, each x row represents the weights coming in the layer n x node
        nn_nodes_n_minus_1 = [len(input_values), *hidden_nodes]
        nn_nodes_n = [*hidden_nodes, output_nodes]
        weights = []
        biases = []
        for layer in range(len(nn_nodes_n)):
            # Xavier weight initialization
            limit = np.sqrt(6 / (nn_nodes_n_minus_1[layer] + nn_nodes_n[layer]))
            weights.append(np.random.uniform(-limit, limit, size=(nn_nodes_n[layer], nn_nodes_n_minus_1[layer])))
            biases.append(np.zeros((nn_nodes_n[layer], 1)))
            

    # Does matrix multiplication of (layer n - 1 to layer n weights) * (layer n - 1 values) and adds layer n biases to get layer n values
    nn_nodes_n = [*hidden_nodes, output_nodes]
    nn_values = [np.array(input_values)]
    activated_nn_values = [np.array(input_values)]

    for layer in range(len(nn_nodes_n)):
        nn_values.append(weights[layer] @ activated_nn_values[layer] + biases[layer])

        # If multiple output nodes (Multi class classification), it leaves the output layer with no activation so it can do softmax activation later
        # If it's a regression problem don't activate the output layer
        if (output_nodes > 1 and layer == len(nn_nodes_n) - 1) or (problem_type == 'Regression' and layer == len(nn_nodes_n) - 1):
            activated_nn_values.append(weights[layer] @ activated_nn_values[layer] + biases[layer])
        
        else:
            activated_nn_values.append(activation_functions.func(weights[layer] @ activated_nn_values[layer] + biases[layer], None, activation_type, 0))
    

    # If mulitple output nodes (Multi class classification), softmax activation turns predicted values to probabilities
    if output_nodes > 1:
        activated_nn_values[-1] = activation_functions.func(None, activated_nn_values[-1], 'Softmax', 0)


    # Output is the final layer values, each column represents a training sample, each row represents an output node
    predicted_values = activated_nn_values[-1]

    return weights, biases, nn_values, activated_nn_values, predicted_values
