#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from torchvision import datasets

def load_mnist_train(train_amount):
    data = datasets.MNIST("../data", train=True, download = True)
    imgs, labels = data.train_data.numpy() , data.train_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_mnist_train: return following shape arrays")
    print("imgs:", return_data[0:train_amount].shape)
    print("labels:", labels[0:train_amount].shape)
    return return_data[0:train_amount]/255.0, labels[0:train_amount]

def load_mnist_test(test_amount):
    data = datasets.MNIST("../data", train=False, download = True)
    imgs, labels = data.test_data.numpy(), data.test_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_mnist_test: return following shape arrays")
    print("imgs:", return_data[0:test_amount].shape)
    print("labels:", labels[0:test_amount].shape)
    return return_data[0:test_amount]/255.0, labels[0:test_amount]

def load_fashion_train(train_amount):
    data = datasets.FashionMNIST("../data", train=True, download = True)
    imgs, labels = data.train_data.numpy() , data.train_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_fashion_train: return following shape arrays")
    print("imgs:", return_data[0:train_amount].shape)
    print("labels:", labels[0:train_amount].shape)
    return return_data[0:train_amount]/255.0, labels[0:train_amount]

def load_fashion_test(test_amount):
    data = datasets.FashionMNIST("../data", train=False, download = True)
    imgs, labels = data.test_data.numpy(), data.test_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_fashion_test: return following shape arrays")
    print("imgs:", return_data[0:test_amount].shape)
    print("labels:", labels[0:test_amount].shape)
    return return_data[0:test_amount]/255.0, labels[0:test_amount]

def load_cifar_train(train_amount):
    data = datasets.CIFAR10("../data", train = True, download = True)
    imgs, labels = data.data, np.array(data.targets)
    print("load_cifar_train: return following shape arrays")
    print("imgs:", imgs[0:train_amount].shape)
    print("labels:", labels[0:train_amount].shape)
    return imgs[0:train_amount]/255.0, labels[0:train_amount]

def load_cifar_test(test_amount):
    data = datasets.CIFAR10("../data", train=False, download = True)
    imgs, labels = data.data, np.array(data.targets)
    print("load_cifar_test: return following shape arrays")
    print("imgs:", imgs[0:test_amount].shape)
    print("labels:", labels[0:test_amount].shape)
    return imgs[0:test_amount]/255.0, labels[0:test_amount]

def load_grimace(img_size, num_test):
    train_labels = np.zeros(18*(20-num_test))
    test_labels = np.zeros(18*num_test)
    train_imgs = np.zeros((int(img_size[0]), int(img_size[1]), 3, 18*(20-num_test)))
    test_imgs = np.zeros((int(img_size[0]), int(img_size[1]), 3, 18*num_test))
    for i in range(18):
        path = "./grimace/dataset/" + str(i) + ".npy"
        subdata = np.load(path)
        for j in range(20):
            if j < num_test:
                test_labels[num_test*i+j] = i
                test_imgs[:, :, :, num_test*i+j] = cv2.resize(subdata[:, :, :, j], (img_size[1], img_size[0]))
            else:
                train_labels[(20-num_test)*i+(j-num_test)] = i
                train_imgs[:, :, :, (20-num_test)*i+(j-num_test)] = cv2.resize(subdata[:, :, :, j], (img_size[1], img_size[0]))
    return np.uint8(train_imgs), train_labels, np.uint8(test_imgs), test_labels
            
from sporco import util,plot

def load_yale(train_amount, test_amount, height, width):
    # 今回はきれいにデータが整理されている２０クラスを使用する
    train_list = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 14, 17, 21, 22, 24, 25, 32, 33, 34, 35]
    # また、学習データに存在しないデータとして１クラスを準備しておく
    unkown_list = 36
    cnt = 0
    train_label = []
    test_label = []
    for i in train_list:
        if cnt == 0:
            tmp = cv2.resize(np.load("../data/Yale/" + str(i) + ".npy"), (width, height))
            train_img = tmp[:, :, 0:train_amount]
            test_img = tmp[:, :, train_amount:]
        else:
            add_array = cv2.resize(np.load("../data/Yale/" + str(i) + ".npy"), (width, height))
            train_img = np.dstack((train_img, add_array[:, :, 0:train_amount]))
            test_img = np.dstack((test_img, add_array[:, :, train_amount:]))
        for j in range(train_amount):
            train_label.append(cnt)
        for j in range(test_amount):
            test_label.append(cnt)
        cnt = cnt + 1
    return train_img, np.array(train_label), test_img, np.array(test_label)