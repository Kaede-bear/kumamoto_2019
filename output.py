from sporco import plot
from sporco import util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def output_Image(flag, Img, file, path):
    if flag == 0:
        fig1 = plot.figure()
        plot.subplot(1, 1, 1)
        plot.imview(util.tiledict(Img[0]), title='Initial Dctionary(Layer1)', fig=fig1)
        fig1.savefig(path + '\\' + file + '(Layer1).png')

        fig2 = plot.figure()
        plot.subplot(1, 1, 1)
        plot.imview(util.tiledict(Img[1]), title='Initial Dctionary(Layer2)', fig=fig2)
        fig2.savefig(path + '\\' + file + '(Layer2).png')
    
    elif flag == 1:
        fig3 = plot.figure()
        for i in range(len(Img)):
            plot.subplot(3, 3, i+1)
            plot.imview(util.tiledict(Img[i]), title='', fig=fig3)
        fig3.savefig(path + '\\' + file + '.png')

    elif flag == 2:
        fig4 = plot.figure()
        for i in range(len(Img)):
            plot.subplot(3, 3, i+1)
            plot.imview(util.tiledict(Img[i]), title='', fig=fig4)
        fig4.savefig(path + '\\' + file + '.png')

    elif flag == 3:
        fig_t = plot.figure()
        for i in range(9):
            plot.subplot(1, 1, 1)
            plot.imview(util.tiledict(Img), title='', fig=fig_t)
        fig_t.savefig(path + '\\' + file + '.png')

def output_Text_conventional(fname, train_amount, accuracy, d_amount, parameter_rho_coef, parameter_rho_dic, parameter_mu, X1_size, X2_size, F1_size, F2_size, L0_1, L0_2, L0_f1, L0_f2, path):
    f = open(path + '\\' + fname,'a')
    f.write("train: " + str(train_amount)
            + "\n[rho_coef: " + str(parameter_rho_coef) + ", rho_dic: " + str(parameter_rho_dic) + ", mu: " + str(parameter_mu) + "]"
            + "\nL0norm(training) [Layer1: " + str(L0_1) + " ({:.2f}".format(100*L0_1/X1_size) + "%)" + ",  Layer2: " + str(L0_2) + " ({:.2f}".format(100*L0_2/X2_size) + "%)" + "]"
            + "\nL0norm(test) [Layer1: " + str(L0_f1) + " ({:.2f}".format(100*L0_f1/F1_size) + "%)" + ",  Layer2: " + str(L0_f2) + " ({:.2f}".format(100*L0_f2/F2_size) + "%)" + "]"
            + "\naccuracy: " + str(accuracy) + "\n\n")
    f.close()

def output_Text_proposed(fname, train_amount, accuracy, d_amount, parameter_rho_coef, parameter_rho_dic, parameter_mu, parameter_gamma, X1_size, X2_size, F1_size, F2_size, L0_1, L0_2, L0_f1, L0_f2, path):
    f = open(path + '\\' + fname,'a')
    f.write("train: " + str(train_amount)
            + "\n[rho_coef: " + str(parameter_rho_coef) + ", rho_dic: " + str(parameter_rho_dic) + ", mu: " + str(parameter_mu) + ", parameter_gamma: " + str(parameter_gamma) + "]"
            + "\nL0norm(training) [Layer1: " + str(L0_1) + " ({:.2f}".format(100*L0_1/X1_size) + "%)" + ",  Layer2: " + str(L0_2) + " ({:.2f}".format(100*L0_2/X2_size) + "%)" + "]"
            + "\nL0norm(test) [Layer1: " + str(L0_f1) + " ({:.2f}".format(100*L0_f1/F1_size) + "%)" + ",  Layer2: " + str(L0_f2) + " ({:.2f}".format(100*L0_f2/F2_size) + "%)" + "]"
            + "\naccuracy: " + str(accuracy) + "\n\n")
    f.close()

def output_Graph(amount, Acc, path):
    x = amount
    y = np.array(Acc).T.tolist()

    fig5 = plt.figure()

    if(type(y[0]) is list):
        for i in range(len(y)):
            ax = fig5.add_subplot(3, 3, i+1)
            ax.plot(x, y[i], color = "red", label = 'Proposed mathod')
    else:
        ax = fig5.add_subplot(1, 1, 1)
        ax.plot(x, y, color = "red", label = 'Proposed mathod')

    ax.set_xlabel('学習枚数')

    plt.ylim([0, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('acc')

    ax.legend(loc='best')

    plt.savefig('figure.png') 
