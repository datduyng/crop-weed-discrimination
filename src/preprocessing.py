import cv2
import glob
import numpy as np


def normalize(x):
    return (x.astype(float) - 128.0) / 128.0

def preprocess():
	grass_path = r'C:\Users\dnguyen52\Downloads\dataset\dataset\grass\*.tif'
	plant_path = r'C:\Users\dnguyen52\Downloads\dataset\dataset\broadleaf\*.tif'

	grass_files = [f for f in glob.glob(grass_path)]
	plant_files = [f for f in glob.glob(plant_path)]
	n_data = len(grass_files) + len(plant_files)

	n_test = 200 
	n_train = n_data - n_test 

	# Image dimension
	im_width = 230
	im_height = 230 
	im_channel = 3

	train_files = [] 
	test_files = [] 
	X_train = np.ndarray(shape=(n_train, im_width, im_height, im_channel), 
						dtype=np.uint8)
	X_test = np.ndarray(shape=(n_test, im_width, im_height, im_channel), 
						dtype=np.uint8)
	Y_train = np.zeros((n_train, 1))
	Y_test = np.zeros((n_test, 1))

	#Split into train set and test set then label it all 
	for idx, _file in enumerate(glob.glob(grass_path)):
	    if(idx < n_test/2): 
	        test_files.append(_file)
	        Y_test[idx] = 0#label image class
	    else: 
	        train_files.append(_file) 
	        Y_train[int(idx - n_test/2)] = 0#label image class


	for idx, _file in enumerate(glob.glob(plant_path)):
	    if(idx < n_test/2):
	        test_files.append(_file)
	        Y_test[idx + int(n_test/2)] = 1#label image class
	    else:
	        train_files.append(_file) 
	        Y_train[int(idx - n_test/2) + len(grass_files)- int(n_test/2)] = 1#label image class

	#Shuffle training index
	train_idxs = np.random.permutation(n_train)
	Y_train  = Y_train[train_idxs]

	for i, idx in enumerate(train_idxs):
		X_train[i] = cv2.resize(cv2.imread(train_files[idx]),(im_width, im_height))
	    
	for idx, _file in enumerate(test_files):
	    X_test[idx] = cv2.resize(cv2.imread(_file),(im_width, im_height))

	return X_train, Y_train, X_test, Y_test