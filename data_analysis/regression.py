import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import requests
import _pickle as pkl
from roi import roi
from random_sample import data_sampling_random
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import csv
import matplotlib.pyplot as plt
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def create_model_fc(input_len):
    # initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    model = Sequential()
    model.add(Input(shape=input_len))
    model.add(Dense(1024, use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('relu'))
    return model


def create_model_cnn(shape):
    model = Sequential()
    model.add(
        Conv2D(64, strides=(1, 1), kernel_size=(1, 2), kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003), data_format='channels_first', input_shape=shape, use_bias=True))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, strides=(1, 1), kernel_size=(1, 2), kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003), data_format='channels_first', use_bias=True))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(1, 3), data_format='channels_first'))

    model.add(Conv2D(256, strides=(1, 1), kernel_size=(1, 2), kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003), data_format='channels_first', use_bias=True))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(256, strides=(1, 1), kernel_size=(1, 2), kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(l2=0.00003), data_format='channels_first', use_bias=True))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(1, 3), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', use_bias=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Activation('relu'))
    return model


def lineNotifyMessage(line_token, msg):
    headers = {
        "Authorization": "Bearer " + line_token,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
    return r.status_code


def data_preparation(dir):
    with open(dir + 'data_train.pkl', 'rb') as f1:
        data_train = pkl.load(f1)
    with open(dir + 'brix_gt_train.pkl', 'rb') as f2:
        label_train = pkl.load(f2)
    with open(dir + 'data_test.pkl', 'rb') as f3:
        data_test = pkl.load(f3)
    with open(dir + 'brix_gt_test.pkl', 'rb') as f4:
        label_test = pkl.load(f4)
    '''
    with open(dir + 'data_train_2.pkl', 'rb') as f5:
        data_train_2 = pkl.load(f5)
    with open(dir + 'data_train_3.pkl', 'rb') as f6:
        data_train_3 = pkl.load(f6)
    with open(dir + 'data_train_4.pkl', 'rb') as f7:
        data_train_4 = pkl.load(f7)
    '''
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
    # data_train = np.concatenate([data_train_1, data_train_2, data_train_3, data_train_4])
    return np.nan_to_num(data_train), np.nan_to_num(data_val), np.nan_to_num(data_test), np.array(
        label_train), np.array(label_val), np.array(
        label_test), brix_gt_train, brix_gt_test, brix_gt_val


def scheduler(epoch):
    print(epoch)
    lr = learning_rate * np.exp(-lr_power * epoch)
    print(lr)
    return lr


def write_csv(dict):
    dir_name = 'checkpoint_data/' + str(date_number) + '/' + model_type.lower() + '_r_' + str(bands[0]) + '_' + str(
        bands[1]) + '/' + str(model_number)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        print("created history saved folder : ", dir_name)
    else:
        print(dir_name, "history saved folder already exists.")
    csv_file_train = dir_name + '/run-train-tag-epoch_loss.csv'
    csv_file_val = dir_name + '/run-val-tag-epoch_loss.csv'
    with open(csv_file_train, 'w') as csvfile_train:
        writer = csv.writer(csvfile_train)
        for x in zip(dict.epoch, dict.history['loss']):
            writer.writerow(x)
    del writer
    with open(csv_file_val, 'w') as csvfile_val:
        writer = csv.writer(csvfile_val)
        for x in zip(dict.epoch, dict.history['val_loss']):
            writer.writerow(x)

def write_result_to_csv(result, number):
    dir_name = 'prediction_result/' + str(date_number) + '/' + model_type.lower() + '_r_' + str(bands[0]) + '_' + str(
        bands[1]) + '/' + str(model_number)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        print("created result saved folder : ", dir_name)
    else:
        print(dir_name, "result saved folder already exists.")
    csv_file = dir_name + "/result.csv"
    with open(csv_file, 'w') as c_f:
        writer = csv.writer(c_f)
        for x in zip(result, number):
            writer.writerow(x)
    del writer

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
        'tai3_part4_train_test_val_v2_10/' + str(bands[0]) + '_' + str(bands[1]) + '/')
    print("Mean Brix: " + str(np.mean(np.concatenate([Y_train, Y_test, Y_val]))))
    print("Std: " + str(np.std(np.concatenate([Y_train, Y_test, Y_val]))))
    pdb.set_trace()
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    if model_type == 'mlp':
        X_train = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_train, axis=1)), axis=1))
        X_test = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_test, axis=1)), axis=1))
        X_val = np.nan_to_num(np.mean(np.nan_to_num(np.mean(X_val, axis=1)), axis=1))
    model_input_shape = X_train[0].shape
    if mode == 'train':
        # preprocessing = data_preprocessing(file_dir)
        # del file_dir
        # preprocessing.data_roi()
        # X_train, Y_train, X_test, Y_test, X_val, Y_val, model_input_shape, brix_gt = preprocessing.data_sampling()

        # Hyper-parameter
        learning_rate = 0.001
        lr_power = 0.00001
        epochs = 3000
        batch_size = 64
        ''''''

        # In case for loading existing model
        # model = load_model('model_regression/tai3_part4_cnn_regression_450_700_1126_4g_best_1.h5')
        if model_type == 'mlp':
            model = create_model_fc(model_input_shape[0])
        else:
            model = create_model_cnn(model_input_shape)  # create new model
        # model_cnn = create_model_fc(model_input_shape)
        pdb.set_trace()
        model.summary()  # get information and structure of model
        # Optimizer
        opt = Adam()
        model.compile(optimizer=opt, loss='mean_squared_logarithmic_error', metrics=["mae"])  # logarithmic

        lr_scheduler = LearningRateScheduler(scheduler)
        save_best = ModelCheckpoint(
            'model_regression/' + str(date_number) + '/tai3_part4_' + model_type.lower() + '_regression_' + str(bands[0]) + '_' + str(
                bands[1]) + '_' + str(date_number) + '_best_' + str(model_number) + '.h5',
            monitor='val_loss', verbose=2,
            save_best_only=True)

        # start model fit
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[save_best, lr_scheduler],
                            validation_data=(X_val, Y_val),
                            verbose=1)

        # save history
        write_csv(history)

    # save final model
    # model.save('model_regression/tai3_part4_cnn_regression_1600_1800_1114_4g_1.h5')
    model = load_model(
        'model_regression/'+str(date_number)+'/tai3_part4_' + model_type.lower() + '_regression_' + str(bands[0]) + '_' + str(
            bands[1]) + '_' + str(date_number) + '_best_' + str(model_number) + '.h5')

    # model = load_model('model_best/regression/'+model_type.lower()+'/'+'tai3_part4_'+model_type+'_regression_'+
    #                   str(bands[0])+'_'+str(bands[1])+'_best.h5')
    model.summary()
    unique_test, counts_test = np.unique(Y_test, return_counts=True)
    #score = model.evaluate(X_test, Y_test, verbose=1)
    #print('Test loss:', score)
    diff_array_0_10 = []
    diff_array_10_11 = []
    diff_array_11_12 = []
    diff_array_12_13 = []
    diff_array_13_14 = []
    diff_array_14_15 = []
    diff_array_15_16 = []
    diff_array_16_17 = []
    diff_array_total = []
    prediction_result = []
    diff_array_total_for_sec = []
    diff_sev_total = []
    for i, x in enumerate(X_test):
        if model_type == 'mlp':
            x = x.reshape(1, model_input_shape[0])  # [0]
        else:
            x = x.reshape(1, model_input_shape[0], model_input_shape[1], model_input_shape[2])
        result = model.predict(x)
        prediction_result.append(result[0])
        diff_temp = np.absolute(np.subtract(result[0], Y_test[i]))
        diff_temp_sev = np.subtract(Y_test[i], result[0])
        diff_sev_total.append(diff_temp_sev)
        diff_array_total.append(diff_temp)
        diff_array_total_for_sec.append(diff_temp*diff_temp)
        if Y_test[i] < 10:
            diff_array_0_10.append(diff_temp)
        elif 10 <= Y_test[i] < 11:
            diff_array_10_11.append(diff_temp)
        elif 11 <= Y_test[i] < 12:
            diff_array_11_12.append(diff_temp)
        elif 12 <= Y_test[i] < 13:
            diff_array_12_13.append(diff_temp)
        elif 13 <= Y_test[i] < 14:
            diff_array_13_14.append(diff_temp)
        elif 14 <= Y_test[i] < 15:
            diff_array_14_15.append(diff_temp)
        elif 15 <= Y_test[i] < 16:
            diff_array_15_16.append(diff_temp)
        elif 16 <= Y_test[i] < 17:
            diff_array_16_17.append(diff_temp)
        print('predtiction : ' + str(result[0]) + ' ground_truth : ' + str(Y_test[i]) + " " + str(i) + " data")
    mean_bias = np.mean(diff_temp_sev)
    mean_sev = np.mean(np.square(np.subtract(diff_sev_total, mean_bias)))
    mean_diff_0_10 = np.mean(diff_array_0_10)
    mean_diff_10_11 = np.mean(diff_array_10_11)
    mean_diff_11_12 = np.mean(diff_array_11_12)
    mean_diff_12_13 = np.mean(diff_array_12_13)
    mean_diff_13_14 = np.mean(diff_array_13_14)
    mean_diff_14_15 = np.mean(diff_array_14_15)
    mean_diff_15_16 = np.mean(diff_array_15_16)
    mean_diff_16_17 = np.mean(diff_array_16_17)
    mean_diff_total = np.mean(diff_array_total)
    mean_diff_sec = np.mean(diff_array_total_for_sec)
    sec = np.sqrt(mean_diff_sec)
    write_result_to_csv([mean_diff_0_10, mean_diff_10_11, mean_diff_11_12, mean_diff_12_13,
                         mean_diff_13_14, mean_diff_14_15, mean_diff_15_16, mean_diff_16_17, mean_diff_total],
                        [len(diff_array_0_10), len(diff_array_10_11), len(diff_array_11_12), len(diff_array_12_13),
                         len(diff_array_13_14), len(diff_array_14_15), len(diff_array_15_16), len(diff_array_16_17),
                         len(diff_array_total)])
    print('mean error for all: ' + str(mean_diff_total) + ' with ' + str(len(diff_array_total)) + ' samples')
    print('mean error for brix 0-10: ' + str(mean_diff_0_10) + ' with ' + str(len(diff_array_0_10)) + ' samples')
    print('mean error for brix 10-11: ' + str(mean_diff_10_11) + ' with ' + str(len(diff_array_10_11)) + ' samples')
    print('mean error for brix 11-12: ' + str(mean_diff_11_12) + ' with ' + str(len(diff_array_11_12)) + ' samples')
    print('mean error for brix 12-13: ' + str(mean_diff_12_13) + ' with ' + str(len(diff_array_12_13)) + ' samples')
    print('mean error for brix 13-14: ' + str(mean_diff_13_14) + ' with ' + str(len(diff_array_13_14)) + ' samples')
    print('mean error for brix 14-15: ' + str(mean_diff_14_15) + ' with ' + str(len(diff_array_14_15)) + ' samples')
    print('mean error for brix 15-16: ' + str(mean_diff_15_16) + ' with ' + str(len(diff_array_15_16)) + ' samples')
    print('mean error for brix 16-17: ' + str(mean_diff_16_17) + ' with ' + str(len(diff_array_16_17)) + ' samples')
    print('sec: '+str(sec))
    print('sev: ' + str(mean_sev))
    #pdb.set_trace()
    #print("Hellow vscode")
    plt.plot(Y_test, prediction_result, 'bo')
    plt.title('Predicted/Measured Result of ' + str(model_type) + ' with bands ' + str(bands[0]) + 'nm-' + str(
        bands[1]) + 'nm')
    plt.xlabel('Measured Brix')
    plt.ylabel('Predicted Brix')
    plt.savefig('figures_result/regression/' + model_type.lower() + '_regression_result_' + str(bands[0]) + '_' + str(
        bands[1]) + '_' + str(date_number) + '_' + str(model_number) + '.png')
    #plt.show()

    # Line notification. Just for reminding me the training is end.
    message = "Training finished! Loss of test is : " + str(mean_diff_total)
    token = 'OyJr9RcOgPdHwwH57xyb7soHBKrJa6lDalaW2aWbd23'
    #lineNotifyMessage(token, message)
