import argparse

import torch
import numpy as np
from random import randint

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar_edited import Cifar10, Cifar100
from scipy.integrate._ivp.radau import predict_factor
from utility.log_edited import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=250, type=int, help="Total number of epochs.") ##default=200
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.0008, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar100(args.batch_size, args.threads)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    PATH = './trained_models/sam_net_250.pth'
    model.load_state_dict(torch.load(PATH))

    predict_all = np.array([])
    correct_all = np.array([], dtype=bool)
    targets_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            rands = torch.clone(targets)
            for r, i in zip(rands.data,range(128)): rands.data[i] = randint(0,9)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, rands)
            correct = torch.argmax(predictions, 1) == rands
            x = targets.cpu().detach().numpy()
            targets_all = np.append(targets_all, x)
            x = predictions.cpu().detach().numpy()
            predict_all = np.append(predict_all, x)
            x = correct.cpu().detach().numpy()
            correct_all = np.append(correct_all, x)

    ##predict_xx = np.concatenate((predict_all, correct_all.T), axis=1)
    predict_all = np.reshape(predict_all, (50000, 10))
    ##targets_all = np.asarray(targets_all).reshape(-1)
    ##correct_all = np.asarray(correct_all).reshape(-1)
    targets_ = np.vstack((targets_all, correct_all)).T
    np.save("preprocessing/c100_train_targ.npy", targets_)
    np.save("preprocessing/c100_train_pred.npy", predict_all)
