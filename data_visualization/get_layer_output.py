from scipy.io import loadmat
import os
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from tensorflow.keras.models import load_model
import random
import _pickle as pkl
import cv2
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tensorflow.keras.models import Model
import brix_labeling
import pdb
import _pickle as pkl

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
wavelength = np.arange(0, 1502)

def get_layer_output(model, x, type):
    # Define the input shape and give the specific number of the layer before the last output layer.
    if type == 'mlp':
        x = x.reshape(1, x.shape[0])
        layer_number=7
    else:
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        layer_number=20
    # Obtain the model
    model = Model(inputs=model.inputs, outputs=model.layers[layer_number].output)
    #model.summary()
    # Output obtained
    layer_output = model.predict(x)
    return layer_output

def data_preparation(dir):
    with open(dir + 'data_train.pkl', 'rb') as f1:
        data_train = pkl.load(f1)
    with open(dir + 'brix_gt_train.pkl', 'rb') as f2:
        label_train = pkl.load(f2)
    with open(dir + 'data_test.pkl', 'rb') as f3:
        data_test = pkl.load(f3)
    with open(dir + 'brix_gt_test.pkl', 'rb') as f4:
        label_test = pkl.load(f4)
    with open(dir + 'data_val.pkl', 'rb') as f8:
        data_val = pkl.load(f8)
    with open(dir + 'label_val.pkl', 'rb') as f9:
        label_val = pkl.load(f9)
    with open(dir + 'brix_gt_train.pkl', 'rb') as f10:
        brix_gt_train = pkl.load(f10)
    with open(dir + 'brix_gt_test.pkl', 'rb') as f11:
        brix_gt_test = pkl.load(f11)
    with open(dir + 'brix_gt_val.pkl', 'rb') as f12:
        brix_gt_val = pkl.load(f12)
    return np.nan_to_num(data_train), np.nan_to_num(data_val), np.nan_to_num(data_test), np.array(
        label_train), np.array(label_val), np.array(
        label_test), brix_gt_train, brix_gt_test, brix_gt_val

def save_to_pkl(numpy_array, name):
    with open("tsne_layer_output/"+name+".pkl", "wb") as f1:
        pkl.dump(numpy_array, f1)
        print("data:"+name)

if __name__ == "__main__":
     # data preparing
    img_rows = 20
    img_cols = 20
    img_depth = 1246
    bands = [400, 1700]
    date_number = 218
    model_number = 5
    model_type = 'cnn'
    mode = 'test' #test
    # data loading
    X_train, X_val, X_test, Y_train, Y_val, Y_test, brix_gt_train, brix_gt_test, brix_gt_val = data_preparation(
        'tai3_part4_train_test_val_v2/' + str(bands[0]) + '_' + str(bands[1]) + '/')
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    # data transforming to 1-D for mlp models
    if model_type == 'mlp':
        X_train = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_train, axis=1)), axis=1))
        X_test = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_test, axis=1)), axis=1))
        X_val = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_val, axis=1)), axis=1))
    model_input_shape = X_train[0].shape
    # laoding the trained models
    model = load_model(
        'model_regression/'+str(date_number)+'/tai3_part4_' + model_type.lower() + '_regression_' + str(bands[0]) + '_' + str(
            bands[1]) + '_' + str(date_number) + '_best_' + str(model_number) + '.h5')
    model.summary()
    # Get the output of all the data
    all_output_matrix = []
    X_all = np.concatenate([X_train, X_val, X_test])
    for i, data in enumerate(X_all):
        all_output_matrix.append(get_layer_output(model, data, model_type))
        print(str(i)+' '+str(all_output_matrix[i].shape))
    # Save the outputs as .pkl file
    save_to_pkl(np.array(all_output_matrix), 'tai3_part4_' + model_type.lower() + '_regression_' + str(bands[0]) + '_' + str(
            bands[1]) + '_' + str(date_number)+ '_' + str(model_number))