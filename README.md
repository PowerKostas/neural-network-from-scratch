# neural-network-from-scratch
The program doesn't have a GUI yet.

<br>

To run doodle/digit recognition:

1. In make_prediction.py, with open(os.path.dirname(os.path.abspath(__file__))+'/x/model.pkl', 'rb') as f:, change x to either doodle_recognition, digit_recognition
2. Go to either doodle_recognition/draw_doodles, digit_recognition/draw_digits and run the file

<br>

To run normal dataset predictions:

- In make_model.py, from x import data_handling, change x to normal_datasets and run the file

<br>

Doodle recognition data downloaded from: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false

Check out file_names in data_handling.py to know which files to download, file_names in data_handling.py and classes in draw_doodles.py need to be updated when downloading other data

<br>

Made in September 2025