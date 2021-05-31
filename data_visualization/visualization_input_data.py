import numpy as np
from sklearn.decomposition import PCA
import _pickle as pkl
import matplotlib.pyplot as plt
from sklearn import manifold

import csv
import pdb


def reduction_dimensionality(data):
    X = np.mean(np.nan_to_num(np.mean(np.nan_to_num(data), axis=1)), axis=1)
    #do pca first, it looks like not necessary
    #pca = PCA(n_components=200) 
    #X_pca = pca.fit_transform(X)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, learning_rate=1000, n_iter=30000, perplexity=20, verbose=2)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm


def labeling(X_rd, label):
    data_classified = [[], [], [], [], []]
    label_ = []
    for i, x in enumerate(label):
        if x < 10:
            data_classified[0].append(X_rd[i])
            label_.append(0)
        elif 10 <= x < 11:
            data_classified[1].append(X_rd[i])
            label_.append(1)
        elif 11 <= x < 12:
            data_classified[2].append(X_rd[i])
            label_.append(2)
        elif 12 <= x < 13:
            data_classified[3].append(X_rd[i])
            label_.append(3)
        elif x >= 13:
            data_classified[4].append(X_rd[i])
            label_.append(4)
    return data_classified, label_

'''
def data_transform(dataset):
    x0, y0 = np.array(dataset[0]).T
    x1, y1 = np.array(dataset[1]).T
    x2, y2 = np.array(dataset[2]).T
    x3, y3 = np.array(dataset[3]).T
    x4, y4 = np.array(dataset[4]).T
    dict = {'data0': [y0, x0], 'data1': [y1, x1], 'data2': [y2, x2], 'data3': [y3, x3], 'data4': [y4, x4]}
    return dict
'''

def read_data(dir, data_name, label_name):
    with open(dir + data_name, 'rb') as f1:
        data = pkl.load(f1)
    with open(dir + label_name, 'rb') as f2:
        label = pkl.load(f2)
    return data, label
'''
def save_to_pkl(list, dir_name):
    with open(dir_name, "wb") as f1:
        pkl.dump(list, f1)
        print("data saved")

def load_pkl(dir_name):
    with open(dir_name, "rb") as f2:
        list = pkl.load(f2)
    return list

def write_error_to_csv(error_list, fn):
    with open(fn, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(error for error in error_list)
'''

if __name__ == "__main__":
    bands_list = [[400,1700], [400,1000], [400,700], [900,1700]]
    accuracy = [[], [], [], []]
    x_label = ['(a)', '(b)', '(c)', '(d)']
    result_list = []
    label_list = []
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 20))
    fig.tight_layout()
    # do dimension reduction of each dataset
    for bands in bands_list:
        dir = 'tai3_part4_train_test_val_v2/'+str(bands[0])+'_'+str(bands[1])+'/'
        data_train, label_train = read_data(dir, 'data_train.pkl', 'label_train.pkl')
        data_test, label_test = read_data(dir, 'data_test.pkl', 'label_test.pkl')
        data_val, label_val = read_data(dir, 'data_val.pkl', 'label_val.pkl')
        X_rd = reduction_dimensionality(np.concatenate([data_train, data_test, data_val]))
        result_list.append(X_rd)
        '''
        X_rd_train = X_rd[:621]
        X_rd_test = np.array(X_rd[621:825])
        X_rd_val = X_rd[825:]
        data_all_classified, label = labeling(X_rd, np.concatenate([label_train, label_test, label_val]))
        Y_rd_train = label[:621]
        Y_rd_test = np.array(label[621:825])
        Y_rd_val = label[825:]
        neigh = KNeighborsClassifier(n_neighbors=1)
        X_train = np.concatenate([X_rd_train, X_rd_val])
        Y_train = np.concatenate([Y_rd_train, Y_rd_val])
        Y_train = Y_train.reshape(Y_train.shape[0], 1)
        Y_test = Y_rd_test.reshape(Y_rd_test.shape[0], 1)

        
        label_list.append(np.concatenate([label_train, label_test, label_val]))
        '''
    k = 0
    for i in range(2):
      for j in range(2):
        points = result_list[k]
        pcm = ax[i][j].scatter(points[:,1], points[:,0], c=label_list[k], cmap='RdBu_r')
        ax[i][j].set_xlabel(x_label[k], fontsize=20)
        k += 1

    handles, labels = ax[0][0].get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(1.11, 0.85), loc='upper right', fontsize=25, markerscale=3)
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label('Brix', fontsize=20, rotation=270)
    fig.savefig('tsne_figures/t-sne_218_3'+'.png', bbox_inches='tight', dpi=150)