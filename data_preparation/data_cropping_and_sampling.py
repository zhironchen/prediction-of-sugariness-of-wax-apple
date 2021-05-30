from random_sample import data_sampling_random
import os
import numpy as np
import brix_labeling
from roi import roi
import _pickle as pkl
import pdb


class data_preprocessing:
    def __init__(self, dir, bands_list):
        self.dir = dir
        self.data_0_10 = []
        self.data_10_11 = []
        self.data_11_12 = []
        self.data_12_13 = []
        self.data_13_ = []
        self.brix_0_10 = []
        self.brix_10_11 = []
        self.brix_11_12 = []
        self.brix_11_12_ground_truth = []
        self.brix_12_13 = []
        self.brix_13_ = []
        self.file_name_0_10 = []
        self.file_name_10_11 = []
        self.file_name_11_12 = []
        self.file_name_12_13 = []
        self.file_name_13_ = []
        self.index_0_10_all = []
        self.index_10_11_all = []
        self.index_11_12_all = []
        self.index_12_13_all = []
        self.index_13__all = []
        self.index_0_10_train = []
        self.index_10_11_train = []
        self.index_11_12_train = []
        self.index_12_13_train = []
        self.index_13__train = []
        self.train_data = []
        self.train_brix = []
        self.index_test_val_brix_0_10 = []
        self.index_test_val_brix_10_11 = []
        self.index_test_val_brix_11_12 = []
        self.index_test_val_brix_12_13 = []
        self.index_test_val_brix_13_ = []
        self.test_brix = []
        self.test_data = []
        self.val_data = []
        self.val_brix = []
        bands_to_index = {400: 4, 700: 34, 900: 54, 1000: 64, 1700: 134} # Index of bands
        self.bands_left = bands_to_index[bands_list[0]]
        self.bands_right = bands_to_index[bands_list[1]]

    def data_roi(self):
        file_name = os.listdir(self.dir + "data/")
        for j, name in enumerate(file_name):
            # loading HSIs data and Brix value
            with open(self.dir + "data/" + name, "rb") as f1:
                data_temp = np.array(pkl.load(f1))[:, :, self.bands_left:self.bands_right]  # :664 1032:1501 #:1501 #1434 90:1457 90:1140
            with open(self.dir + "brix/" + name, "rb") as f2:
                brix_temp = np.array(pkl.load(f2))
            # x y for data cropping criteria
            x, y, z = data_temp.shape
            y_min_for_two_cols = 50
            # to define the max pieces this sample can be cut into(for both upper and bottom part)
            if x >= 160:
                max_p = 4
            elif 120 <= x < 160:
                max_p = 3
            else:
                max_p = 2
            # extract data
            if brix_temp < 10:
                for i in range(0, max_p):
                    if y <= y_min_for_two_cols:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_0_10.append(d1_0)
                        del d1_0
                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_0_10.append(u1_0)
                        del u1_0
                        # append brix n times
                        for n in range(0, 2):
                            self.brix_0_10.append(brix_temp)
                    else:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), 0, int(img_cols))
                        self.data_0_10.append(d1_0)
                        del d1_0
                        d2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_0_10.append(d2_0)
                        del d2_0

                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), 0, int(img_cols))
                        self.data_0_10.append(u1_0)
                        del u1_0
                        u2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_0_10.append(u2_0)
                        del u2_0

                        for n in range(0, 4):
                            # append brix
                            self.brix_0_10.append(brix_temp)
                self.file_name_0_10.append(name)
            elif 10 <= brix_temp < 11:
                for i in range(0, max_p):
                    if y <= y_min_for_two_cols:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_10_11.append(d1_0)
                        del d1_0
                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_10_11.append(u1_0)
                        del u1_0
                        # append brix n times
                        for n in range(0, 2):
                            self.brix_10_11.append(brix_temp)
                    else:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), 0, int(img_cols))
                        self.data_10_11.append(d1_0)
                        del d1_0
                        d2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_10_11.append(d2_0)
                        del d2_0

                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), 0, int(img_cols))
                        self.data_10_11.append(u1_0)
                        del u1_0
                        u2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_10_11.append(u2_0)
                        del u2_0
                        for n in range(0, 4):
                            # append brix
                            self.brix_10_11.append(brix_temp)
                self.file_name_10_11.append(name)
            elif 11 <= brix_temp < 12:
                for i in range(0, max_p):
                    if y <= y_min_for_two_cols:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_11_12.append(d1_0)
                        del d1_0
                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_11_12.append(u1_0)
                        del u1_0
                        # append brix n times
                        for n in range(0, 2):
                            self.brix_11_12.append(brix_temp)
                    else:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), 0, int(img_cols))
                        self.data_11_12.append(d1_0)
                        del d1_0
                        d2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_11_12.append(d2_0)
                        del d2_0

                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), 0, int(img_cols))
                        self.data_11_12.append(u1_0)
                        del u1_0
                        u2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_11_12.append(u2_0)
                        del u2_0
                        for n in range(0, 4):
                            # append brix
                            self.brix_11_12.append(brix_temp)
                self.file_name_11_12.append(name)
            elif 12 <= brix_temp < 13:
                for i in range(0, max_p):
                    if y <= y_min_for_two_cols:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_12_13.append(d1_0)
                        del d1_0
                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_12_13.append(u1_0)
                        del u1_0
                        # append brix n times
                        for n in range(0, 2):
                            self.brix_12_13.append(brix_temp)
                    else:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), 0, int(img_cols))
                        self.data_12_13.append(d1_0)
                        del d1_0
                        d2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_12_13.append(d2_0)
                        del d2_0

                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), 0, int(img_cols))
                        self.data_12_13.append(u1_0)
                        del u1_0
                        u2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_12_13.append(u2_0)
                        del u2_0

                        for n in range(0, 4):
                            # append brix
                            self.brix_12_13.append(brix_temp)
                self.file_name_12_13.append(name)
            elif 13 <= brix_temp:
                for i in range(0, max_p):
                    if y <= y_min_for_two_cols:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_13_.append(d1_0)
                        del d1_0
                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), -int(img_cols / 2),
                                   int(img_cols / 2))
                        self.data_13_.append(u1_0)
                        del u1_0
                        # append brix n times
                        for n in range(0, 2):
                            self.brix_13_.append(brix_temp)
                    else:
                        # for down part
                        d1_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), 0, int(img_cols))
                        self.data_13_.append(d1_0)
                        del d1_0
                        d2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_13_.append(d2_0)
                        del d2_0

                        # for upper part
                        u1_0 = roi(data_temp, int(i * img_rows), int((i + 1) * img_rows), 0, int(img_cols))
                        self.data_13_.append(u1_0)
                        del u1_0
                        u2_0 = roi(data_temp, -int((i + 1) * img_rows), -int(i * img_rows), -int(img_cols), 0)
                        self.data_13_.append(u2_0)
                        del u2_0

                        for n in range(0, 4):
                            # append brix
                            self.brix_13_.append(brix_temp)
                self.file_name_13_.append(name)
            del brix_temp, data_temp
        # count samples of each groups
        print('debug')

    def data_sampling(self):
        ## Random sampling of data into 5 groups. We tried to make a balanced dataset as possible.

        # All the data obtain from random sampling of the original dataset
        (X_0_10_all, Y_0_10_all) = data_sampling_random(self.data_0_10, self.brix_0_10, 218, keep_left=False)  # number of data : 218
        (X_10_11_all, Y_10_11_all) = data_sampling_random(self.data_10_11, self.brix_10_11, 162, keep_left=False)  # 162
        (X_11_12_all, Y_11_12_all) = data_sampling_random(self.data_11_12, self.brix_11_12, 218, keep_left=False)
        (X_12_13_all, Y_12_13_all) = data_sampling_random(self.data_12_13, self.brix_12_13, 218, keep_left=False)  # 218
        (X_13_all, Y_13_all) = data_sampling_random(self.data_13_, self.brix_13_, 218, keep_left=False)
        del self.data_0_10, self.data_11_12, self.data_12_13, self.data_13_
        # Sampling into train, test and validation set.
        (X_0_10_train, Y_0_10_train, X_0_10_val_test, Y_0_10_val_test) = data_sampling_random(X_0_10_all, Y_0_10_all,
                                                                                              131,
                                                                                              keep_left=True)  # 131
        (X_10_11_train, Y_10_11_train, X_10_11_val_test, Y_10_11_val_test) = data_sampling_random(X_10_11_all,
                                                                                                  Y_10_11_all, 97, #97
                                                                                                  keep_left=True)
        (X_11_12_train, Y_11_12_train, X_11_12_val_test, Y_11_12_val_test) = data_sampling_random(X_11_12_all,
                                                                                                  Y_11_12_all, 131,
                                                                                                  keep_left=True)
        (X_12_13_train, Y_12_13_train, X_12_13_val_test, Y_12_13_val_test) = data_sampling_random(X_12_13_all,
                                                                                                  Y_12_13_all, 131,
                                                                                                  keep_left=True)  # 131
        (X_13_train, Y_13_train, X_13_val_test, Y_13_val_test) = data_sampling_random(X_13_all, Y_13_all, 131,
                                                                                      keep_left=True)
        (X_0_10_test, Y_0_10_test, X_0_10_val, Y_0_10_val) = data_sampling_random(X_0_10_val_test, Y_0_10_val_test, 43,
                                                                                  keep_left=True)  # 43
        (X_10_11_test, Y_10_11_test, X_10_11_val, Y_10_11_val) = data_sampling_random(X_10_11_val_test,
                                                                                      Y_10_11_val_test, 32,
                                                                                      keep_left=True)
        (X_11_12_test, Y_11_12_test, X_11_12_val, Y_11_12_val) = data_sampling_random(X_11_12_val_test,
                                                                                      Y_11_12_val_test, 43,
                                                                                      keep_left=True)
        (X_12_13_test, Y_12_13_test, X_12_13_val, Y_12_13_val) = data_sampling_random(X_12_13_val_test,
                                                                                      Y_12_13_val_test, 43,
                                                                                      keep_left=True) 
        (X_13_test, Y_13_test, X_13_val, Y_13_val) = data_sampling_random(X_13_val_test, Y_13_val_test, 43,
                                                                          keep_left=True)
        X_train = np.array(X_0_10_train + X_10_11_train + X_11_12_train + X_12_13_train + X_13_train)
        train_brix = Y_0_10_train + Y_10_11_train + Y_11_12_train + Y_12_13_train + Y_13_train
        Y_train = train_brix
        X_test = np.array(X_0_10_test + X_10_11_test + X_11_12_test + X_12_13_test + X_13_test)
        self.test_brix = Y_0_10_test + Y_10_11_test + Y_11_12_test + Y_12_13_test + Y_13_test
        Y_test = self.test_brix
        X_val = np.array(X_0_10_val + X_10_11_val + X_11_12_val + X_12_13_val + X_13_val)
        val_brix = Y_0_10_val + Y_10_11_val + Y_11_12_val + Y_12_13_val + Y_13_val
        Y_val = val_brix

        input_shape = X_train[0].shape

        return X_train, Y_train, X_test, Y_test, X_val, Y_val, input_shape, self.test_brix, val_brix, train_brix


def save_dataset(data_train, label_train, data_test, label_test, data_val, label_val, brix_gt_test, brix_gt_val,
                 brix_gt_train, dir):
    with open(dir + "data_train" + ".pkl", "wb") as f1:
        pkl.dump(data_train, f1)
    with open(dir + "label_train" + ".pkl", "wb") as f2:
        pkl.dump(label_train, f2)
    with open(dir + "data_test" + ".pkl", "wb") as f3:
        pkl.dump(data_test, f3)
    with open(dir + "label_test" + ".pkl", "wb") as f4:
        pkl.dump(label_test, f4)
    with open(dir + "data_val" + ".pkl", "wb") as f5:
        pkl.dump(data_val, f5)
    with open(dir + "label_val" + ".pkl", "wb") as f6:
        pkl.dump(label_val, f6)
    with open(dir + "brix_gt_test.pkl", "wb") as f8:
        pkl.dump(brix_gt_test, f8)
    with open(dir + "brix_gt_train.pkl", "wb") as f9:
        pkl.dump(brix_gt_train, f9)
    with open(dir + "brix_gt_val.pkl", "wb") as f10:
        pkl.dump(brix_gt_val, f10)


if __name__ == "__main__":
    # data preparing
    img_rows = 20
    img_cols = 20
    img_depth = 1246
    bands = [400, 1700]
    all_pickle_dir = 'location/of/all/pickle/data'
    save_dir = 'location/to/save/pre-processed/pkl/file' + str(bands[0]) + '_' + str(bands[1]) + '/'
    preprocessing = data_preprocessing(all_pickle_dir, bands)
    del all_pickle_dir
    preprocessing.data_roi()
    X_train, Y_train, X_test, Y_test, X_val, Y_val, model_input_shape, brix_gt_test, brix_gt_val, brix_gt_train = preprocessing.data_sampling()
    save_dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val, brix_gt_test, brix_gt_val, brix_gt_train, save_dir)
    print("done")
