import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import _pickle as pkl
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.pyplot import legend

from sklearn.metrics import mean_absolute_error
import csv
import pdb


def reduction_dimensionality(data):
    X = np.reshape(data, (data.shape[0], data.shape[2]))
    pca = PCA(n_components=200)
    X_pca = pca.fit_transform(X)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, learning_rate=1000, n_iter=30000, perplexity=20, verbose=2)#0.01 3000 30 10
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm

def load_label(dir, label_name):
    with open(dir + label_name, 'rb') as f1:
        label = pkl.load(f1)
    return label

def load_data(data_name):
    with open("tsne_layer_output/"+data_name+".pkl", 'rb') as f1:
        label = pkl.load(f1)
    return label


if __name__ == "__main__":
    model_type = 'mlp'
    date_number = 218
    if model_type == 'mlp':
        bands_list = [[400,1700,5], [400,1000,5], [400,700,2], [900,1700,2]]
    else:
        bands_list = [[400,1700,5], [400,1000,3], [400,700,3], [900,1700,2]]
    accuracy = [[], [], [], []]
    x_label = ['(a)', '(b)', '(c)', '(d)']
    result_list = []
    label_list = []
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 20))
    fig.tight_layout()
    
    for bands in bands_list:
        label_dir = 'tai3_part4_train_test_val_v2/'+str(bands[0])+'_'+str(bands[1])+'/'
        data_name = 'tai3_part4_' + model_type.lower() + '_regression_' + str(bands[0]) + '_' + str(
            bands[1]) + '_' + str(date_number)+ '_' + str(bands[2])
        label_train = load_label(label_dir, 'label_train.pkl')
        label_test = load_label(label_dir, 'label_test.pkl')
        label_val = load_label(label_dir, 'label_val.pkl')
        data = load_data(data_name)
        X_rd = reduction_dimensionality(data)
        label_list.append(np.concatenate([label_train, label_test, label_val]))
        result_list.append(X_rd)

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
    fig.savefig('tsne_figures/t-sne_218_after_'+model_type+'_regression.png', bbox_inches='tight', dpi=150)