import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class ThreeLayerNet:
    # 初期化関数 NNのパラメータの初期値設定を行う
    def __init__(self, size = {'in': 10, 'hidden1': 10, 'hidden2': 10, 'out': 10},
                 weight_init_std=0.01, experiment={'init_param': False, 'func': False}):
        self.params = {}
        self.experiment = experiment
        # 重みの初期化
        # experiment[0]がTrueの時に、重みの初期値を全てゼロにする
        # paramsはNNの重みとバイアスを保持するインスタンス変数
        if self.experiment['init_param'] == False:
            self.params['W1'] = weight_init_std * np.random.randn(size['in'], size['hidden1'])
            self.params['b1'] = np.zeros(size['hidden1'])
            self.params['W2'] = weight_init_std * np.random.randn(size['hidden1'], size['hidden2'])
            self.params['b2'] = np.zeros(size['hidden2'])
            self.params['W3'] = weight_init_std * np.random.randn(size['hidden2'], size['out'])
            self.params['b3'] = np.zeros(size['out'])
        else:
            self.params['W1'] = weight_init_std * np.zeros((size['in'], size['hidden1']))
            self.params['b1'] = np.zeros(size['hidden1'])
            self.params['W2'] = weight_init_std * np.zeros((size['hidden1'], size['hidden2']))
            self.params['b2'] = np.zeros(size['hidden2'])
            self.params['W3'] = weight_init_std * np.zeros((size['hidden2'], size['out']))
            self.params['b3'] = np.zeros(size['out'])
    
    # 推論を行う
    def predict(self, input_data):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        a1 = np.dot(input_data, W1) + b1
        if self.experiment['func'] == False:
            z1=sigmoid(a1)
        else:
            z1=a1
        a2 = np.dot(z1, W2) + b2
        if self.experiment['func'] == False:
            z2=sigmoid(a2)
        else:
            z2=a2
        a3 = np.dot(z2, W3) + b3
        output_data = softmax(a3)
        
        return output_data
    
    # 損失関数の値を求める
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    # 認識精度
    def accuracy(self, x, t):
        y = self.predict(x)
        # argmaxは配列内の最大要素の位置を返す
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 重みパラメータに対する勾配を求める
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
            
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        
        return grads
    
    # 高速化版　次の章の内容含む
    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        if self.experiment['func'] == False:
            z1=sigmoid(a1)
        else:
            z1=a1
        a2 = np.dot(z1, W2) + b2
        if self.experiment['func'] == False:
            z2=sigmoid(a2)
        else:
            z2=a2
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        
        # backward
        dy = (y - t) / batch_num
        
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
      
        dz2 = np.dot(dy, W3.T)
        da2 = sigmoid_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(da2, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads