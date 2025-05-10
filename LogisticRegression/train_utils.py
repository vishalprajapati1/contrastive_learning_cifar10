import numpy as np
from typing import Tuple

from LogisticRegression.model import LinearModel
from utils import get_data_batch


def calculate_loss(model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False) -> float:
    y_preds = model(X).squeeze()
    if is_binary:
        epsilon = 1e-6
        # clipping to avoid log(0) which is undefined
        y_preds = np.clip(y_preds, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_preds) + (1 - y) * np.log(1 - y_preds)) # binary cross-entropy loss
    else:
        loss = 0
        num_classes = y_preds.shape[1]
        for truth, p_predection in zip(y, y_preds):
            loss -= 1 * np.log(p_predection[truth])
        loss = loss / len(y)
    return loss



def calculate_accuracy(model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False) -> float:
    y_preds = model(X).squeeze()
    if is_binary:
        acc = np.mean((y_preds > 0.5) == y) # binary classification accuracy
    else:
        acc = np.mean(np.argmax(y_preds, axis=1) == y)
    return acc


def evaluate_model(model: LinearModel, X: np.ndarray, y: np.ndarray,
                   batch_size: int, is_binary: bool = False) -> Tuple[float, float]:

    acc, loss = 0.0, 0.0
    cnt = 0
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        acc += calculate_accuracy(model, X_batch, y_batch, is_binary)
        loss += calculate_loss(model, X_batch, y_batch, is_binary)
        cnt += 1
    return loss / cnt , acc / cnt


def fit_model(model: LinearModel, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray, num_iters: int,
             lr: float, batch_size: int, l2_lambda: float,
             grad_norm_clip: float, is_binary: bool = False) -> Tuple[list, list, list, list]:

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for i in range(num_iters + 1):

        X_batch, y_batch = get_data_batch(X_train, y_train, batch_size)
        y_preds = model(X_batch).squeeze() # added .squeeze() as a correction.
        
        # calculate loss
        loss = calculate_loss(model, X_batch, y_batch, is_binary)
        
        # calculate accuracy
        acc = calculate_accuracy(model, X_batch, y_batch, is_binary)
        
        # calculate gradient
        if is_binary:
            grad_W = ((y_preds - y_batch) @ X_batch).reshape(-1, 1)
            grad_b = np.mean(y_preds - y_batch)
        else:
            # Compute gradient for multinomial logistic regression

            num_classes = 10 # TODO: hoist this as a constant
            one_hot = np.zeros((y_batch.size, num_classes))
            for j, y in enumerate(y_batch): one_hot[j, y] = 1
            grad_W = (y_preds - one_hot).T.dot(X_batch).T
            grad_b = np.mean(y_preds - one_hot, axis=0)
            
        # regularization
        grad_W += l2_lambda * model.W
        grad_b += l2_lambda * model.b

        # cliping grad_norm
        grad_norm = np.linalg.norm(grad_W) + np.linalg.norm(grad_b)
        if grad_norm > grad_norm_clip:
            grad_W = grad_W * (grad_norm_clip / grad_norm)
            grad_b = grad_b * (grad_norm_clip / grad_norm)

        # perfoming Stochastic gradient descent
        model.W -= lr * grad_W
        model.b -= lr * grad_b
        
        if i % 10 == 0:
            train_losses.append(loss)
            train_accs.append(acc)

            val_loss, val_acc = evaluate_model(
                model, X_val, y_val, batch_size, is_binary)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Iter {i}/{num_iters} - Train Loss: {loss:.4f} - Train Acc: {acc:.4f}'
                  f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

    return train_losses, train_accs, val_losses, val_accs
