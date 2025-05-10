from argparse import ArgumentParser

from utils import *
from LogisticRegression.model import LogisticRegression, SoftmaxRegression
from LogisticRegression.train_utils import fit_model, evaluate_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'seed', type=int, default=1,
        help='seed')
    parser.add_argument(
        '--train_data_path', type=str, default="data/cifar10_train.npz",
        help='Path to the training data')
    parser.add_argument(
        '--test_data_path', type=str, default="data/test_images.npz",
        help='Path to the test data')
    parser.add_argument(
        '--num_iters', type=int, default=1000,
        help='Number of iterations for training')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for training')
    parser.add_argument(
        '--mode', type=str, default='logistic', choices=
        ['logistic', 'softmax', 'cont_rep', 'fine_tune_linear', 'fine_tune_nn', 'retrain'],
        help='Mode of operation'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256, # deault is 256
        help='Batch size for training')
    parser.add_argument(
        '--l2_lambda', type=float, default=0.1,
        help='L2 regularization for training')
    parser.add_argument(
        '--grad_norm_clip', type=float, default=4.0, 
        help='Clip gradient norm')
    # Contrastive Representation Learning specific arguments
    parser.add_argument(
        '--z_dim', type=int, default=128,
        help='Representation dimension for the encoder')
    parser.add_argument(
        '--encoder_path', type=str, default='models/encoder.pth',
        help='Path to save the encoder model')
    args = parser.parse_args()
    args.test_data_path = args.test_data_path[:-4] + f'_{args.sr_no}.npz'

    np.random.seed(args.seed)

    if args.mode == 'logistic' or args.mode == 'softmax':
        from LogisticRegression.main import main
    elif args.mode == 'cont_rep' or args.mode == 'fine_tune_linear' or args.mode == 'fine_tune_nn' or args.mode == 'retrain':
        from ContrastiveRepresentation.main import main
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    main(args)
