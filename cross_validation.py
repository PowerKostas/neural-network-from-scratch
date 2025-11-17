import numpy as np

def start(input_values, true_values, k):
    # Splits data into 80% train and validation and 20% test
    split_percentage = 0.8
    train_val_input_values = input_values[:, :int(input_values.shape[1] * split_percentage)]
    train_val_true_values = true_values[:, :int(input_values.shape[1] * split_percentage)]
    test_input_values = input_values[:, int(input_values.shape[1] * split_percentage):]
    test_true_values = true_values[:, int(input_values.shape[1] * split_percentage):]


    # Shuffles indices of the train and validation samples 
    train_val_indices = np.arange(train_val_input_values.shape[1])
    np.random.shuffle(train_val_indices)


    # Splits train and validation indices into k folds
    fold_sizes = [train_val_indices.shape[0] // k] * k
    for i in range(train_val_indices.shape[0] % k):
        fold_sizes[i] += 1

    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        fold_indices.append(train_val_indices[current:current + fold_size])
        current += fold_size

    return train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices
