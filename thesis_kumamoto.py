#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from load_dataset import load_mnist_test, load_mnist_train
from load_dataset import load_fashion_test, load_fashion_train
#from conduct_svm import svm
import Convert as con
import ConvDL as cdl
import conduct_svm as svm
import conduct_pooling as cp
import output as op
import random

#実験のオプションを設定
def make_option():
    opt = {}
    # 訓練画像の枚数
    opt['train_amount'] = [9]
    #opt['train_amount'] = [50, 100]
    # テスト画像の枚数
    opt['test_amount'] = 500
    # 最大繰り返し回数
    opt['iteration'] = [1, 100, 3]
    # 更新終了の閾値
    opt['thr'] = 0.001
    #学習の層の数
    opt['Learning'] = 2
    #フィルターのサイズ
    opt['d1_size, d2_size'] = [6, 6]
    #フィルターの数
    opt['d_amount'] = [4, 10]
    #パラメータ設定
    opt['rho_coef'] = 0.2
    opt['rho_dic_conventional'] = 100.0 #係数の射影
    opt['rho_dic_propposed'] = 1.0 #係数の射影
    opt['mu'] = 1.0 #勾配法
    opt['gamma_conventional'] = 1.0  #近似誤差(L1)
    opt['gamma_proposed'] = 1.0 #近似誤差(L1)
    #実験結果の保存先
    opt['file_path'] = 'C:\\CDL\\result'
    return opt

#辞書の初期値設定
def dictionary_set(learning, d_size, d_amount):
    D = []
    for i in range(learning):
        D.append(np.random.normal(0, 1, (d_size[i], d_size[i], d_amount[i])))
    return D

opt = make_option()
Learned_D1 = []
Learned_D2 = []

D0 = dictionary_set(opt['Learning'], opt['d1_size, d2_size'], opt['d_amount'])
op.output_Image(0, D0, 'Initial_Dictionary', opt['file_path'])

fname = 'result_conventional.txt'
#fname = 'result_proposed.txt'

for train_amount in opt['train_amount']:
    #opt['rho_coef'] = train_amount * 0.001

    #train_data, train_label = load_mnist_train(train_amount)
    #test_data, test_label = load_mnist_test(opt["test_amount"])

    train_data, train_label = load_fashion_train(train_amount)
    test_data, test_label = load_fashion_test(opt["test_amount"])

    '''for i in range(int(train_amount/100)):
        outlier = random.randrange(0, train_amount, 1)
        train_data[outlier] = 0'''

    train_data = train_data.transpose(1,2,0)
    test_data = test_data.transpose(1,2,0)
    op.output_Image(3, train_data, 'Fashion-MNIST', opt['file_path'])

    print("\nTrain data: " + str(train_amount) + "\n")

    time_start = time.time()

    print("\n[gamma: %1.2f, rho_dic: %1.2f, mu: %1.2f]" % (opt['gamma_conventional'], opt['rho_dic_conventional'], opt['mu']))
    #print("\n[rho_coef: %1.2f, rho_dic: %1.2f, mu: %1.2f, gamma: %1.2f]" % (opt['rho_coef'], opt['rho_dic_propposed'], opt['mu'], opt['gamma']))

    D1, X1, L0_1 = cdl.coef_dic_update_L2_L0(opt['gamma_conventional'], opt['rho_dic_conventional'], opt['mu'], train_data, D0[0], opt['iteration'], opt['thr'], 'Layer1')            
    #D1, X1, L0_1 = cdl.coef_dic_update_L1_L0(opt['rho_coef'], opt['rho_dic_propposed'], opt['mu'], opt['gamma'], train_data, D0[0], opt['iteration'], opt['thr'], 'Layer1')
    if(L0_1 == 0):
        continue

    X1 = X1.transpose(2, 3, 0, 1)
    Xp1 = cp.pooling_layer(X1, train_amount)
    Xp1 = Xp1.transpose(1, 2, 0)

    D2, X2, L0_2 = cdl.coef_dic_update_L2_L0(opt['gamma_conventional'], opt['rho_dic_conventional'], opt['mu'], Xp1, D0[1], opt['iteration'], opt['thr'], 'Layer2')
    #D2, X2, L0_2 = cdl.coef_dic_update_L1_L0(opt['rho_coef'], opt['rho_dic_propposed'], opt['mu'], opt['gamma'], Xp1, D0[1], opt['iteration'], opt['thr'], 'Layer2')
    if(L0_2 == 0):
        continue

    train_F1, L0_train1 = cdl.feature_extraction_L2_L0(opt['gamma_conventional'], opt['mu'], train_data, D1, opt['iteration'], opt['thr'], 'Feature1')
    #train_F1, L0_train1 = cdl.feature_extraction_L1_L0(opt['rho_coef'], opt['gamma'], opt['mu'], train_data, D1, opt['iteration'], opt['thr'], 'Feature1')
    if(L0_train1 == 0):
        continue

    train_F1 = train_F1.transpose(2, 3, 0, 1)
    train_F1_P = cp.pooling_layer(train_F1, train_amount)
    train_F1_P = train_F1_P.transpose(1, 2, 0)

    train_F2, L0_train2 = cdl.feature_extraction_L2_L0(opt['gamma_conventional'], opt['mu'], train_F1_P, D2, opt['iteration'], opt['thr'], 'Feature2')
    #train_F2, L0_train2 = cdl.feature_extraction_L1_L0(opt['rho_coef'], opt['gamma'], opt['mu'], train_F1_P, D2, opt['iteration'], opt['thr'], 'Feature2')
    if(L0_train2 == 0):
        continue

    train_F2 = train_F2.transpose(2, 3, 0, 1)
    train_F2_P = cp.pooling_layer2(train_F2, train_amount)

    #train_F2 = train_F2.transpose(0,1,3,2)
    #train_F2 = train_F2.reshape(-1,train_F2.shape[3])

    clf = svm.train_svm(train_F2_P, train_label)

    test_F1, L0_test1 = cdl.feature_extraction_L2_L0(opt['gamma_conventional'], opt['mu'], test_data, D1, opt['iteration'], opt['thr'], 'Feature1')
    #test_F1, L0_test1 = cdl.feature_extraction_L1_L0(opt['rho_coef'], opt['gamma'], opt['mu'], test_data, D1, opt['iteration'], opt['thr'], 'Feature1')
    if(L0_test1 == 0):
        continue

    test_F1 = test_F1.transpose(2, 3, 0, 1)
    test_F1_P = cp.pooling_layer(test_F1, opt['test_amount'])
    test_F1_P = test_F1_P.transpose(1, 2, 0)

    test_F2, L0_test2 = cdl.feature_extraction_L2_L0(opt['rho_coef'], opt['mu'], test_F1_P, D2, opt['iteration'], opt['thr'], 'Feature2')
    #test_F2, L0_test2 = cdl.feature_extraction_L1_L0(opt['rho_coef'], opt['gamma'], opt['mu'], test_F1_P, D2, opt['iteration'], opt['thr'], 'Feature2')
    if(L0_test2 == 0):
        continue

    test_F2 = test_F2.transpose(2, 3, 0, 1)
    test_F2_P = cp.pooling_layer2(test_F2, opt['test_amount'])

    #test_F2 = test_F2.transpose(0,1,3,2)
    #test_F2 = test_F2.reshape(-1,test_F2.shape[3])

    accuracy = svm.test_svm(clf, test_F2_P, test_label)

    time_end = time.time()

    print("accuracy: %1.3f" % accuracy)
    print("solve time: %.2fs\n" % (time_end-time_start))

    Learned_D1.append(D1)
    Learned_D2.append(D2)

    op.output_Text_conventional(fname, train_amount, accuracy, opt['d_amount'], opt['gamma_conventional'], opt['rho_dic_conventional'], opt['mu'], train_F1.size, train_F2.size, test_F1.size, test_F2.size, L0_train1, L0_train2, L0_test1, L0_test2, opt['file_path'])
    #op.output_Text_proposed(fname, train_amount, accuracy, opt['d_amount'], opt['rho_coef'], opt['rho_dic_propposed'], opt['mu'], opt['gamma'], train_F1.size, train_F2.size, test_F1.size, test_F2.size, L0_train1, L0_train2, L0_test1, L0_test2, opt['file_path'])

#op.output_Image(1, Learned_D1, 'Learned_Dictionary_Conventional(Layer1)', opt['file_path'])
#op.output_Image(2, Learned_D2, 'Learned_Dictionary_Conventional(Layer2)', opt['file_path'])
op.output_Image(1, Learned_D1, 'Learned_Dictionary_Proposed(Layer1)', opt['file_path'])
op.output_Image(2, Learned_D2, 'Learned_Dictionary_Proposed(Layer2)', opt['file_path'])
Learned_D1.clear()
Learned_D2.clear()

