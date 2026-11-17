import numpy as np
import os

# Doing this to import module from parent folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_validation


def get_input_true_values(k):
    # Reads the data, does one hot encoding for the true values, normalizes input values, shuffles and transposes both input and true values
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
    file_names = ['full_numpy_bitmap_apple.npy', 'full_numpy_bitmap_car.npy', 'full_numpy_bitmap_crown.npy', 
                  'full_numpy_bitmap_flower.npy', 'full_numpy_bitmap_fork.npy', 'full_numpy_bitmap_hand.npy',
                  'full_numpy_bitmap_headphones.npy', 'full_numpy_bitmap_keyboard.npy', 'full_numpy_bitmap_light bulb.npy',
                  'full_numpy_bitmap_moustache.npy', 'full_numpy_bitmap_radio.npy', 'full_numpy_bitmap_rifle.npy', 
                  'full_numpy_bitmap_stairs.npy', 'full_numpy_bitmap_stop sign.npy', 'full_numpy_bitmap_tree.npy']
    input_values = []
    true_values = []

    for id, file_name in enumerate(file_names):
        file_path = os.path.join(data_dir, file_name)

        temp_input_values = np.load(file_path) 
        temp_input_values = temp_input_values[:20000]       
        input_values.append(temp_input_values)

        temp_true_values = np.zeros((temp_input_values.shape[0], len(file_names)), dtype = np.float32)
        temp_true_values[:, id] = 1.0
        true_values.append(temp_true_values)

    input_values = np.vstack(input_values)  
    input_values = input_values.astype(np.float32) / 255.0
    true_values = np.vstack(true_values)

    permutation= np.random.permutation(len(input_values))
    input_values = input_values[permutation]
    true_values = true_values[permutation]

    input_values = input_values.T
    true_values = true_values.T


    # Cross validation
    train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices = cross_validation.start(input_values, true_values, k)
    return train_val_input_values, train_val_true_values, test_input_values, test_true_values, fold_indices
