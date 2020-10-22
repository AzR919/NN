import sys
import os
import numpy as np
import argparse
import model
import matplotlib.pyplot as plt

import ipdb



if __name__=="__main__":

    parser = argparse.ArgumentParser("Argument parser for SGD")
    parser.add_argument("--train_root", type=str, default='../training_data_student/train.csv', help="path to the training data")
    parser.add_argument("--test_root", type=str, default='../training_data_student/test.csv', help="path to the test data")
    parser.add_argument("--params_root", type=str, default=None, help="path to the initial weights")

    parser.add_argument("--num_epochs", type=int, default=15, help='number of epochs to train for')
    parser.add_argument("--step_size", type=float, default=0.01, help='step size for SGD')
    parser.add_argument("--hidden_dim", type=int, default=256, help='dimension for the hidden layer')
    parser.add_argument("--seed", type=int, default=0, help='seed for randomization')
    parser.add_argument("--eps", type=float, default=1e-8, help='eps to prevent negative log')

    args = parser.parse_args()

    if (not os.path.exists(args.train_root)):
            print("Train data path does not exist")
            exit(1)
    if (not os.path.exists(args.test_root)):
            print("Test data path does not exist")
            exit(1)

    #ipdb.set_trace()

    print("-----------------------------------------")
    print("Start Loading Data")
    print("-----------------------------------------")

    train_raw = np.loadtxt(open(args.train_root, "rb"), delimiter=",")
    train_f = train_raw[:,:-1]
    train_l = train_raw[:,-1]

    test_raw = np.loadtxt(open(args.test_root, "rb"), delimiter=",")
    test_f = test_raw[:,:-1]
    test_l = test_raw[:,-1]

    if args.params_root != None :
        alpha_1 = np.loadtxt(open(os.path.join(args.params_root, "alpha1.txt"), "rb"), delimiter=",")
        alpha_2 = np.loadtxt(open(os.path.join(args.params_root, "alpha2.txt"), "rb"), delimiter=",")
        beta_1 = np.loadtxt(open(os.path.join(args.params_root, "beta1.txt"), "rb"), delimiter=",")
        beta_2 = np.loadtxt(open(os.path.join(args.params_root, "beta2.txt"), "rb"), delimiter=",")

        model = model.nn_model(args, train_f.shape, True,alpha_1, alpha_2, beta_1, beta_2)
    else:
        model = model.nn_model(args, train_f.shape)
    
    
    print("-----------------------------------------")
    print("Finish Loading Data")
    print("-----------------------------------------")

    train_losses, test_losses, test_accs = model.train(train_f, train_l, test_f, test_l)

    title = "Loss, num_epochs: {}, step_size: {}, hidden_size: {}".format(args.num_epochs, args.step_size, args.hidden_dim)
    if args.params_root == None:
        title = title + ", random _init"

    plt.plot(train_losses, label="Train_loss")
    plt.plot(test_losses, label="Test_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title(title)
    plt.show()

    #ipdb.set_trace()
    
    

    print("-----------------------------------------")
    print("All Done")
    print("-----------------------------------------")