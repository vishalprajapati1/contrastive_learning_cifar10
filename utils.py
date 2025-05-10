import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import ContrastiveRepresentation.pytorch_utils as ptu

def get_data(data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
             is_binary: bool = False, grayscale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    X = X.transpose(0, 3, 1, 2)
    if is_binary:
        # in binary case: we are just taking the first two classes: airplanes and automobiles
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # Rescaling
    X = X / 255.0

    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert test_ratio < 1 and test_ratio > 0

    num_test_samples = int(len(X) * test_ratio)

    indices = np.random.permutation(len(X))
    indices = np.random.permutation(indices)
    X_train = X[indices[num_test_samples:]]
    y_train = y[indices[num_test_samples:]]
    X_test = X[indices[:num_test_samples]]
    y_test = y[indices[:num_test_samples]]

    return X_train, y_train, X_test, y_test


def get_data_batch(X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.random.choice(len(X), size=batch_size, replace=False)
    return X[idxs], y[idxs]

def get_contrastive_data_batch(X: np.ndarray, y: np.ndarray, batch_size: int):  
    with torch.no_grad():
        try:
            y_temp = ptu.to_numpy(y)
        except:
            y_temp = y
    cache = dict()
    print("Caching the indices for each class")
    for i in range(0, 10):
        cache[i] = np.where(y_temp == i)[0]

    while True:
        anc_idx = []
        pos_idx = []
        neg_idx = []
        for _ in range(batch_size):
            anchor_class = np.random.randint(0, 10)
            anc_idx.append(np.random.choice(cache[anchor_class]))
            pos_idx.append(np.random.choice(cache[anchor_class]))
            neg_class = np.random.randint(0, 10)
            while neg_class == anchor_class: neg_class = np.random.randint(0, 10)
            neg_idx.append(np.random.choice(cache[neg_class]))
        yield  X[anc_idx], X[pos_idx], X[neg_idx]

def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    with torch.no_grad():
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.savefig('images/loss.png')
        plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    with torch.no_grad():
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend()
        plt.savefig('images/acc.png')
        plt.close()


def plot_tsne(
       z: torch.Tensor, y: torch.Tensor 
) -> None:

    mapping = { 0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck',
    }

    y = [mapping[i.item()] for i in y]

    with torch.no_grad():
        print("We are now plotting the t-SNE, it'll take a while.")
        z_tsne = TSNE(n_components=2).fit_transform(z)
        import pandas as pd
        fig, ax = plt.subplots()
        groups = pd.DataFrame(z_tsne, columns=['x', 'y']).assign(category=y).groupby('category')
        for name, points in groups:
            ax.scatter(points.x, points.y, label=name)
        ax.legend()
        fig.savefig('images/tsne.png')

    return
