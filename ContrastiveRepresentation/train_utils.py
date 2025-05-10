import torch
from argparse import Namespace
from typing import Union, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_data, train_test_split

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
import torch
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy

def calculate_loss(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
 the model
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(y_logits, y)
    return loss

def retrain(model_path, encoder):
    encoder.load_state_dict(torch.load(model_path))
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    loss_fn = TripletLoss()
    X, y = get_data()
    X, y, _, _ = train_test_split(X, y)
    encoder.to(ptu.device)
    losses = []
    sample_size = 100
    sampler = get_contrastive_data_batch(X, y, sample_size)
    num_iters = 2
    for i in range(num_iters):
        print("we are now retraining the given model")
        print(f"started iteration {i}")
        x_a, x_p, x_n = next(sampler)

        x_a = ptu.from_numpy(x_a).float()
        x_p = ptu.from_numpy(x_p).float()
        x_n = ptu.from_numpy(x_n).float()

        encoder_optimizer.zero_grad()

        z_a = encoder(x_a)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        loss = loss_fn(z_a, z_p, z_n)
        loss.backward()
        encoder_optimizer.step()
        print(loss.item())
        losses.append(loss.item())
    torch.save(encoder.state_dict(), 'models/encoder_retrained.pth')
    return losses


def calculate_accuracy(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    _, predicted_labels = torch.max(y_logits, dim=1)
    correct_predictions = (predicted_labels == y).sum().item()
    total_predictions = y.size(0)
    acc = correct_predictions / total_predictions
    return acc

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        anchor_pos_dist_L2 = (anchor-pos).pow(2).sum(1)
        anchor_neg_dist_L2 = (anchor-neg).pow(2).sum(1)
        loss = F.relu(anchor_pos_dist_L2-anchor_neg_dist_L2+self.margin).mean()
        return loss

def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 10000,
        batch_size: int = 1024,
        learning_rate: float = 1e-3) -> None:
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_fn = TripletLoss()
    encoder.to(ptu.device)
    losses = []
    sampler = get_contrastive_data_batch(X, y, batch_size)

    for i in range(num_iters):

        x_a, x_p, x_n = next(sampler)
        x_a = ptu.from_numpy(x_a).float()
        x_p = ptu.from_numpy(x_p).float()
        x_n = ptu.from_numpy(x_n).float()

        encoder_optimizer.zero_grad()

        z_a = encoder(x_a)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        loss = loss_fn(z_a, z_p, z_n)
        loss.backward()
        encoder_optimizer.step()
        print(loss.item())

        losses.append(loss.item())

    return losses

# def evaluate_model(
#         encoder: torch.nn.Module,
#         classifier: Union[LinearModel, torch.nn.Module],
#         X: torch.Tensor,
#         y: torch.Tensor,
#         batch_size: int = 256,
#         is_linear: bool = False
# ) -> Tuple[float, float]:
#     '''
#     Evaluate the model on the given data.

#     Args:
#     - encoder: torch.nn.Module, the encoder model
#     - classifier: Union[LinearModel, torch.nn.Module], the classifier model
#     - X: torch.Tensor, images
#     - y: torch.Tensor, labels
#     - batch_size: int, batch size for evaluation
#     - is_linear: bool, whether the classifier is linear

#     Returns:
#     - loss: float, loss of the model
#     - acc: float, accuracy of the model
#     '''
#     raise NotImplementedError('Get the embeddings from the encoder and pass it to the classifier for evaluation')

#     # HINT: use calculate_loss and calculate_accuracy functions for NN classifier and calculate_linear_loss and calculate_linear_accuracy functions for linear (softmax) classifier

#     # return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
) -> Tuple[List[float], List[float], List[float], List[float]]:

    if args.mode == 'fine_tune_linear':
        X_train = encoder(X_train)
        X_val = encoder(X_val)
        X_train = X_train.detach().numpy()
        y_train = y_train.detach().numpy()
        X_val = X_val.detach().numpy()
        y_val = y_val.detach().numpy()
        return fit_linear_model(classifier, X_train, y_train, X_val, y_val, args.num_iters, args.lr, args.batch_size, args.l2_lambda, args.grad_norm_clip, is_binary=False)
    else:
        X_train = encoder(X_train)
        X_val = encoder(X_val)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        num_iters = 2000
        batch_size = 32
        for i in range(num_iters):
            with torch.no_grad():
                X_batch, y_batch = get_data_batch(X_train, y_train, batch_size)
            y_preds = classifier(X_batch).squeeze() 
            loss = loss_fn(y_preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(calculate_accuracy(y_preds, y_batch))
            val_accs.append(calculate_accuracy(classifier(X_val), y_val))
            val_losses.append(calculate_loss(classifier(X_val), y_val))
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{num_iters}: Train Loss = {train_losses[-1]}, Train Accuracy = {train_accs[-1]}, Validation Loss = {val_losses[-1]}, Validation Accuracy = {val_accs[-1]}")

        return train_losses, train_accs, val_losses, val_accs
