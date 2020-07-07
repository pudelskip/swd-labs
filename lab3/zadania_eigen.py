#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig

    eigvec = np.linalg.eig(A)[1]
    eigvec = np.transpose(eigvec)

    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec)


def EVD_decomposition(A):
    eigval, normals = np.linalg.eig(A)
    L = np.zeros((len(eigval),len(eigval)))

    for l,e in enumerate(eigval):
        L[l,l]=e

    print("L:\n", L)
    K = normals
    print("K:\n", K)
    K_1 = np.linalg.inv(K)
    print("K^-1:\n", K_1)
    A = np.matmul(np.matmul(K,L),K_1)
    print("A:\n",A)



def plot_attractors(A, vectors):
    colors = ['red','orange','blue','cyan',"yellow",'purple','lime','deeppink','maroon','khaki']


    eigvec = np.linalg.eig(A)[1]
    eigvec1 = np.transpose(eigvec)
    to_keep =[]
    for i in range(len(eigvec1)):
        keep = True
        for j in range(i+1,len(eigvec1)):
            if 1-cosine(eigvec1[i], eigvec1[j])>0.99:
                keep=False
                break
        if keep:
            to_keep.append(eigvec1[i])

    eigvec1 = np.array(to_keep)
    eigvec2 = -1*(eigvec1)
    eigvec = np.concatenate([eigvec1,eigvec2],axis=0)


    # eigvec =eigvec2

    attractors = {}

    for i,v in enumerate(eigvec):
        attractors[tuple(v)]=colors[i]

    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        transformed_vec= v
        for _ in range(10):
            transformed_vec = A.dot(transformed_vec)
            transformed_vec = transformed_vec / np.linalg.norm(transformed_vec)


        attractor= None
        sim = None
        for a in attractors.keys():
            if sim is None:
                sim = 1 - cosine(transformed_vec, a)
                attractor = a

                continue
            new_sim = 1-cosine(transformed_vec, a)

            if new_sim > sim:
                attractor = a
                sim=new_sim

        self_sim = 1-cosine(transformed_vec, v)
        if self_sim > sim:
            v = v / np.linalg.norm(v)
            plt.quiver(0.0, 0.0, v[0], v[1], width=0.004, color="black", scale_units='xy', angles='xy',
                       scale=1,
                       zorder=4)
        else:
            v = v / np.linalg.norm(v)
            plt.quiver(0.0, 0.0, v[0], v[1], width=0.004, color=attractors[attractor], scale_units='xy', angles='xy', scale=1,
                       zorder=4)


    for i, v in enumerate(eigvec):
        # Plot original vector.
        v = v / np.linalg.norm(v)
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color=colors[i], scale_units='xy', angles='xy', scale=1,
                   zorder=4)



    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.margins(0.05)


    plt.show()
    pass


def test_A1(vectors):
    """Standard scaling transformation."""
    A = np.array([[2, 0],
                  [0, 2]])
    visualize_transformation(A, vectors)


def test_A2(vectors):
    A = np.array([[-1, 2],
                  [2, 1]])
    visualize_transformation(A, vectors)


def test_A3(vectors):
    A = np.array([[3, 1],
                  [0, 2]])
    visualize_transformation(A, vectors)



def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)