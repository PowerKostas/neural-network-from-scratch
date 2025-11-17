import numpy as np
import cost_functions
import feed_forward
import backprop
from normal_datasets import data_handling # from options: doodle_recognition, digit_recognition, normal_datasets, depending on the problem
import pickle

# Training data, the input should be an array of features for each sample, the true values should be an array of labels for each sample
#input_values = np.array([[0.2, 0.12, 0.15], [0.8, 0.9, 0.65], [0.13, 0.27, 0.05], [0, 0, 0]]).T # Each row represents a feature, each column represents a sample
#true_values = np.array([[1], [0], [1]]).T # Each row represents a feature, each column represents a sample

k = 5 # Number of folds
train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices = data_handling.get_input_true_values(k)


# Parameters
problem_type = 'Regression' # Problem options: Binary, Regression (binary/categorical output or number output)
visual_cost_type = 'CCE' # Try different cost types for better understanding, for example, MSE gives skewed results when there are many outliers
hidden_nodes = [32] # How many nodes each hidden layer has
practical_cost_type = 'MSE' # Options: MAE, MSE, BCE (Only for binary problems)
activation_type = 'ReLU' # Options: Sigmoid (Only for binary problems), Tanh (Only for binary problems), ReLU
lr = 0.00002 # Typical options: 1, 0.1, 0.01, 0.001, 0.0001, 0.00001 ...
epochs = 750


# If multiple output nodes (Multi class classification), CCE is mandatory in my program
if train_val_true_values.shape[0] > 1: 
    practical_cost_type = 'CCE'


val_costs = []
for fold in range(k):
    print(f'Fold {fold+1}:\n')

    # Does k fold cross validation, there are k folds, in each fold it uses 1 split for validation and all the rest for training
    val_indices = fold_indices[fold]
    train_indices = np.concatenate([fold_indices[i] for i in range(k) if i != fold])

    train_input_values = train_val_input_values[:, train_indices]
    train_true_values = train_val_true_values[:, train_indices]
    val_input_values = train_val_input_values[:, val_indices]
    val_true_values = train_val_true_values[:, val_indices]


    # Training
    weights, biases = [], []
    temp_cost = 0.1
    for epoch in range(epochs):
        weights, biases, nn_values, activated_nn_values, predicted_values = feed_forward.start(train_input_values, train_true_values.shape[0], hidden_nodes, activation_type, weights, biases, problem_type)
        cost = cost_functions.func(train_true_values, predicted_values, visual_cost_type, 0)

        if visual_cost_type == 'MSE': 
            cost = np.sqrt(cost) # Gets RMSE cost

        print(f'Epoch {epoch + 1}: {cost}')

        #if abs(temp_cost - cost) / temp_cost < 0.00001: # Percentage early stopping
            #break
        #temp_cost = cost

        weights, biases = backprop.start(weights, biases, train_true_values, activated_nn_values, nn_values, practical_cost_type, activation_type, lr, problem_type)


    # Validation
    _, _, _, _, predicted_values = feed_forward.start(val_input_values, val_true_values.shape[0], hidden_nodes, activation_type, weights, biases, problem_type)
    cost = cost_functions.func(val_true_values, predicted_values, visual_cost_type, 0)

    if visual_cost_type == 'MSE':
        cost = np.sqrt(cost)

    val_costs.append(cost)

    print(f'\nFold {fold + 1} validation cost: {cost}\n')

print(f'Average fold validation cost: {np.mean(val_costs)}\n')


# Testing
_, _, _, _, predicted_values = feed_forward.start(test_input_values, test_true_values.shape[0], hidden_nodes, activation_type, weights, biases, problem_type)
cost = cost_functions.func(test_true_values, predicted_values, visual_cost_type, 0)

if visual_cost_type == 'MSE': 
    cost = np.sqrt(cost) # Gets RMSE cost

print(f'Testing cost: {cost}\n')


# Analysing the testing results, if it's a binary problem, it counts the correct predictions, if it's a regression problem, it compares results
# with predicting the average every time
if problem_type == 'Binary':
    if practical_cost_type == 'BCE': # Only 2 labels per sample (0 or 1), so it compares predictions with 0.5
        test_true_values_labels = test_true_values
        predicted_values_labels = (predicted_values >= 0.5).astype(int)
        length = test_true_values_labels.shape[1]
    
    elif practical_cost_type == 'CCE': # More than 2 labels per sample, so it gets the biggest prediction
        test_true_values_labels = np.argmax(test_true_values, axis=0)
        predicted_values_labels = np.argmax(predicted_values, axis=0)
        length = len(test_true_values_labels)

    accurate_predictions = np.sum(test_true_values_labels == predicted_values_labels)
    print(f'Accuracy: {accurate_predictions}/{length}')
    print(f'Accuracy Percentage: {(accurate_predictions / length) * 100}%\n')

else:
    avg_cost = cost_functions.func(test_true_values, test_true_values.mean(), visual_cost_type, 0)

    if visual_cost_type == 'MSE': 
        avg_cost = np.sqrt(avg_cost) # Gets RMSE cost

    print(f'Percentage difference with predicting the average: {((avg_cost - cost) / avg_cost) * 100}%\n')


# Save model info so I don't have to train it every time I make predictions, only use it if training takes too long, it gets saved on Desktop, 
# move it in the folder it needs to be
model = {'hidden_nodes': hidden_nodes, 'activation_type': activation_type, 'weights': weights, 'biases': biases, 'problem_type': problem_type}

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
