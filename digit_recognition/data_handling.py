import numpy as np
import os

# Doing this to import module from parent folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_validation


def get_input_true_values(k):
    # Reads the data
    with np.load(os.path.dirname(os.path.abspath(__file__))+'/mnist.npz') as data:
        input_values = data['training_images'][:, :, 0].T
        true_values = data['training_labels'][:, :, 0].T


    # Cross validation
    train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices = cross_validation.start(input_values, true_values, k)
    return train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices
