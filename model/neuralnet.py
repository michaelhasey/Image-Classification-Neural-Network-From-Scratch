# 10-601_HW5_feature.py
# Michael Hasey
# mhasey@andrew.cmu.edu
# mhasey


# ***********************************************************
# NEURAL NETWORK ALGORITHM
# ***********************************************************


# ******************************************************************************
# Imports
# ******************************************************************************

import sys
import csv
import numpy as np
from numpy import genfromtxt
import copy


# ******************************************************************************
# Helper Functions
# ******************************************************************************

# 1. convert csv to np array ***************************************************

def csv_to_npy(csv_to_convert):
    np_array = genfromtxt(csv_to_convert, delimiter=',')
    label_array = np_array[:, 0]
    label_array = label_array.astype(int)
    np_array_bias = copy.deepcopy(np_array)
    np_array_bias[:, 0] = 1
    return np_array_bias, label_array

# 2. create label & feature np array *******************************************

def data_array_prep(raw_dataset):
    features, labels = csv_to_npy(raw_dataset)
    # one hot encode label array, "np.eye" does this
    classes = 4
    labels = np.eye(classes)[labels]
    return labels, features

# 3. initialize weights ********************************************************

def init_weights(train_features, units, flag):
    classes = 4
    # +1 indicates extra bias col
    alpha_rows = int(units)
    alpha_cols = int(train_features.shape[1])
    beta_rows = classes
    beta_cols = units + 1
    if flag == 1:
        alpha_init = np.random.uniform(-0.1,0.1,size=(alpha_rows,alpha_cols))
        beta_init = np.random.uniform(-0.1,0.1,size=(beta_rows,beta_cols))
    if flag == 2:
        alpha_init = np.zeros((alpha_rows,alpha_cols))
        beta_init = np.zeros((beta_rows,beta_cols))
    return alpha_init, beta_init

# 4. Loss - Mean Cross Entropy *************************************************

def MCEL(features_array, labels_array, alpha, beta):
    losses = []
    for row in range(0,features_array.shape[0]):
        x = features_array[row]
        y = labels_array[row]
        # pre-activation layer 1 
        a = np.matmul(np.transpose(x), np.transpose(alpha))
        # hidden activation layer (sigmoid)
        z = 1 / (1 + np.exp(-1 * a))
        # add bias to z
        z_width = int(z.shape[0]) + 1
        z_hat = np.ones(z_width)
        z_hat[1:] = z
        # pre-activation layer 2
        b = np.matmul(np.transpose(z_hat), np.transpose(beta))
        # prediction layer (softmax)
        y_hat = np.exp(b) / np.sum(np.exp(b))
        # loss
        l = -1 * (np.sum(np.matmul(y, np.transpose(np.log(y_hat)))))
        losses.append(l)
    loss_average = sum(losses) / len(losses)
    return loss_average

# 4. Error *********************************************************************

def error_rate(features_array, labels_array, alpha, beta):
    
    total_errors = 0
    total_predictions = 0
    predictions = []

    for row in range(0,features_array.shape[0]):
        total_predictions += 1
        x = features_array[row]
        y = labels_array[row]
        # pre-activation layer 1 
        a = np.matmul(np.transpose(x), np.transpose(alpha))
        # hidden activation layer (sigmoid)
        z = 1 / (1 + np.exp(-1 * a))
        # add bias to z
        z_width = int(z.shape[0]) + 1
        z_hat = np.ones(z_width)
        z_hat[1:] = z
        # pre-activation layer 2
        b = np.matmul(np.transpose(z_hat), np.transpose(beta))
        # prediction layer (softmax)
        y_hat = np.exp(b) / np.sum(np.exp(b))
        # record predictionn
        pred_idx = np.argmax(y_hat)
        predictions.append(pred_idx)
        # error
        if np.argmax(y) != np.argmax(y_hat):
            total_errors += 1

    error = total_errors / total_predictions
    error = format(error, '.6f')
    return error, predictions
    

# ******************************************************************************
# Command Line Arguments
# ******************************************************************************

if __name__ == '__main__':
    # input files
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    # output files
    train_out = sys.argv[3]
    valid_out = sys.argv[4]
    metrics_out = sys.argv[5]
    # hyperparameters
    num_epoch = sys.argv[6]
    hidden_units = int(sys.argv[7])
    init_flag = sys.argv[8]
    lr = sys.argv[9]

num_epoch = int(num_epoch)
hidden_units = int(hidden_units)
init_flag = int(init_flag)
lr = float(lr)


# ******************************************************************************
# Data Prep
# ******************************************************************************

train_labels, train_feats = data_array_prep(train_input)
valid_labels, valid_feats = data_array_prep(valid_input)


# ******************************************************************************
# Initialization
# ******************************************************************************

# weightsexit
alpha_init_yes, beta_init_yes = init_weights(train_feats, hidden_units, init_flag)
alpha = alpha_init_yes
beta = beta_init_yes

# adagrad  matrices
s_alpha = np.zeros((alpha.shape[0], alpha.shape[1]))
s_beta = np.zeros((beta.shape[0], beta.shape[1]))


# ******************************************************************************
# Training
# ******************************************************************************

epoch_list = []
losses = []
losses_train = []
losses_valid = []
train_error = []
valid_error = []

for epoch in range(0, int(num_epoch)):
    epoch_list.append(epoch)
    loss = []
    for row in range(0,train_labels.shape[0]):
        x = train_feats[row]
        y_real = train_labels[row]

        # Forward Pass

        # pre-activation layer 1 
        a = np.matmul(np.transpose(x), np.transpose(alpha))
        # hidden activation layer (sigmoid)
        z = 1 / (1 + np.exp(-1 * a))
        # add bias to z
        z_width = int(z.shape[0]) + 1
        z_hat = np.ones(z_width)
        z_hat[1:] = z
        # pre-activation layer 2
        b = np.matmul(np.transpose(z_hat), np.transpose(beta))
        # prediction layer (softmax)
        y_hat = np.exp(b) / np.sum(np.exp(b))
        # loss
        l = -1 * (np.sum(np.matmul(y_real, np.transpose(np.log(y_hat)))))

        # Backward Pass

        # pre-activation layer 2 derivs
        b = y_hat - y_real
        # beta derivs
        b_dim = np.array([b])
        z_hat = np.array([z_hat])
        beta_deriv = np.matmul(np.transpose(b_dim), z_hat)
        # hidden layer derivs
        z_hat = np.matmul(b, beta)
        # remove bias from hidden layer derivs
        z = np.array([z_hat[1:]])
        # pre-activation layer 1 derivs
        sig = (1/(1 + np.exp(-1 * a)))
        a = z * (sig * (1-sig))
        a = a[0]
        # alpha derivs
        a_dim = np.array([a])
        x_dim = np.array([x])
        alpha_deriv = np.matmul(np.transpose(a_dim), x_dim)

        # update weights - sgd - adagrad
        s_alpha = s_alpha + (np.multiply(alpha_deriv, alpha_deriv))
        #alpha = alpha - (np.multiply((lr / np.sqrt(s_alpha + 0.00001)), alpha_deriv))
        alpha = alpha - (np.multiply((lr / np.sqrt(s_alpha + 0.00001)), alpha_deriv))
        s_beta = s_beta + (np.multiply(beta_deriv, beta_deriv))
        # beta = beta - (np.multiply((lr / np.sqrt(s_beta + 0.00001)), beta_deriv))
        beta = beta - (np.multiply((lr / np.sqrt(s_beta + 0.00001)), beta_deriv))

    # loss - cross entropy 
    train_loss = MCEL(train_feats, train_labels, alpha, beta)
    valid_loss = MCEL(valid_feats, valid_labels, alpha, beta)
    losses_train.append(train_loss)
    losses_valid.append(valid_loss)
    # print(f'epoch={epoch} crossentropy(train): {train_loss}')
    # print(f'epoch={epoch} crossentropy(validation): {valid_loss}')
    losses.append(f'epoch={epoch} crossentropy(train): {train_loss}')
    losses.append(f'epoch={epoch} crossentropy(validation): {valid_loss}')


# ******************************************************************************
# Testing
# ******************************************************************************

train_error, train_predictions = error_rate(train_feats, train_labels, alpha, beta)
valid_error, valid_predictions = error_rate(valid_feats, valid_labels, alpha, beta)
losses.append(f'error(train): {train_error}')
losses.append(f'error(validation): {valid_error}')
# print(f'error(train): {train_error}')
# print(f'error(validation): {valid_error}')


# ******************************************************************************
# Record Results
# ******************************************************************************

# train labels
with open(train_out, 'w+') as writer_train_label:
    for elem in train_predictions:
        writer_train_label.write(str(elem) + '\n')
# valid labels
with open(valid_out, 'w+') as writer_valid_label:
    for elem in valid_predictions:
        writer_valid_label.write(str(elem) + '\n')
# losses
with open(metrics_out, 'w+') as writer_metrics_out:
    for elem in losses:
        writer_metrics_out.write(str(elem) + '\n')