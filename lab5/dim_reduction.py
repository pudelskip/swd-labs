from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
from numpy.linalg import *
from numpy.random import *
from scipy.spatial import distance_matrix




class MDS_classic():

    def __init__(self, n_components):
        self.n_components = n_components

    def mds_transform(self, d, dimensions=2):
        """
        Multidimensional Scaling - Given a matrix of interpoint distances,
        find a set of low dimensional points that have similar interpoint
        distances.

        Source: https://github.com/stober/mds
        """
        E = (-0.5 * d ** 2)

        # Use mat to get column and row means to act as column and row means.
        Er = mat(mean(E, 1))
        Es = mat(mean(E, 0))

        # From Principles of Multivariate Analysis: A User's Perspective (page 107).
        F = array(E - transpose(Er) - Es + mean(E))

        [U, S, _] = svd(F)

        Y = U * sqrt(S)

        return Y[:, 0:dimensions]

    def norm(self,vec):
        return sqrt(sum(vec ** 2))

    def square_points(self, size):
        nsensors = size ** 2
        return array([(i / size, i % size) for i in range(nsensors)])

    def fit_transform(self, data):
        if isinstance(data, pd.DataFrame):
            d= distance_matrix(data.values, data.values)
        else:
            d = distance_matrix(data, data)
        return self.mds_transform(d,self.n_components)


N_COMPONENTS = 2

mds = MDS_classic(N_COMPONENTS)
tsne = TSNE(N_COMPONENTS)
isomap = Isomap(n_components=N_COMPONENTS, n_neighbors=3)
lle= LocallyLinearEmbedding(n_components=N_COMPONENTS, method='standard')


data_dir = './dr_data'

datasets = ['cars.csv', 'Mordhau_Weapons.csv']
methods = {"MDS classic": mds, "TSNE": tsne,"ISOMAP - k=3": isomap,"LLE - standard": lle}

def read_dataset(name, label_column):
    data = pd.read_csv(name, header=None)

    if isinstance(label_column, int):
        data.reset_index(inplace=True,drop=True)
        labels = data.iloc[:, label_column]
        data = data.drop(columns=[data.columns[label_column]])
    else:
        labels = data[label_column]
        data = data.drop(columns=[label_column])

    return labels, data

def reduce_dimensions(method, data):
    transformed = method.fit_transform(data)
    return transformed

def plot_data(axes, transformed, index, method_name, labels=None):
    ax = sns.scatterplot(transformed[:, 0], transformed[:, 1], legend=False, ax=axes[index[0], index[1]])
    ax.set_title(method_name)
    if labels is not None:
        for i in range(len(transformed)):
            ax.annotate(labels[i],
                         (transformed[i, 0], transformed[i, 1]),
                         textcoords="offset points",
                         xytext=(0, 0),
                         ha='left')


if __name__ == '__main__':
    sns.set(font_scale=0.8)
    for file_name in datasets:
        f, axes = plt.subplots(2, 2, figsize=(12, 16))

        ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        sns.despine(left=True)
        labels, data = read_dataset(os.path.join(data_dir,  file_name), 0)
        for i, (name, m) in enumerate(methods.items()):
            transformed = m.fit_transform(data)
            plot_data(axes, transformed, ax_indices[i], name, labels)
        plt.savefig(f"{file_name}.png", dpi=300)
        plt.show()

    swiss_roll = make_swiss_roll()
    f, axes = plt.subplots(2, 2, figsize=(8, 12))

    ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    sns.despine(left=True)
    for i, (name, m) in enumerate(methods.items()):
        data, _ = swiss_roll
        transformed = m.fit_transform(data)
        plot_data(axes, transformed, ax_indices[i], name)

    plt.savefig("swiss.png", dpi=300)
    plt.show()
