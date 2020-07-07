import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def evd(M, symetric=False):
    return np.linalg.eigh(M) if symetric else np.linalg.eig(M)

def pseudo_invert(M):
    M = np.vectorize(lambda x: 1/x if x!=0 else 0)(M)
    if M.shape[0] != M.shape[1]:
        M = M.T
    return M

def compress(M, method='custom',k=None):

    if method == 'custom':
        u, t, v = simple_svd(M)
    elif method == 'library':
        u, t, v = np.linalg.svd(M, full_matrices=True)
        t = create_t_matrix(u.shape[0], v.shape[0], t)
    else:
        print("Unsupported method using default custom")
        u, t, v = simple_svd(M)

    if k is not None:
        u, t, v = truncate_svd(u, t, v, k)
    out = np.dot(u, np.dot(t, v))

    return out



def simple_svd(M):

    C = np.matmul(M.T, M)
    eigval, eigvec = evd(C)

    sorted_indices = np.argsort(eigval, axis=0)[::-1]
    eigval = eigval[sorted_indices]
    eigvec = eigvec[:, sorted_indices]

    E_sq = np.diag(eigval)
    h_pad = M.shape[0] - E_sq.shape[0]
    v_pad = M.shape[1] - E_sq.shape[1]

    E = E_sq.copy()
    E = np.pad(E, ((0, h_pad), (0, 0)), 'constant') if h_pad >= 0 else E_sq[:h_pad, :]
    E = np.pad(E, ((0, 0), (0, v_pad)), 'constant') if v_pad >= 0 else E_sq[:, :v_pad]
    T = np.sqrt(E)

    V = eigvec
    T_1 = pseudo_invert(T)
    U = np.matmul(np.matmul(M, V), T_1)

    return U, T, V.T


def create_t_matrix(height, width, eigenvalues):
    t = np.zeros((height, width))
    smaller_dim = min(height, width)
    t[:smaller_dim, :smaller_dim] = np.diag(eigenvalues)
    return t


def truncate_svd(u, t, v, k):
    t_trunc = t[:k,:k]
    u_trunc = u[:,:k]
    v_trunc = v[:k,:]
    return u_trunc, t_trunc, v_trunc


def process_image(image, method, k):

    assert len(image.shape) in [2,3], "Image should be 2D or 3D matrix"

    if len(image.shape)==3:
        channels = []
        for ch in range(image.shape[2]):
            single_channel = image[...,ch]
            channels.append(compress(single_channel,method=method,k=k))
        compressed = np.stack(channels,axis=-1)
    else:
        compressed = compress(image, method=method,k=k)

    return compressed.astype(float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', help="Path to image file", required=True)
    parser.add_argument('-out', help="Path to save image")
    parser.add_argument('-svd', help="Implementation", default='custom')
    parser.add_argument('-k', help="Number of singular values", type=int)
    args = parser.parse_args()

    image = np.array(Image.open(args.f), np.float)/255
    compressed = process_image(image,args.svd, args.k)
    if args.out is not None:
        out_image = Image.fromarray(np.uint8(compressed*255))
        out_image.save(args.out)
    else:
        plt.imshow(compressed)
        plt.show()
