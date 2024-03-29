import numpy as np
import random
from keras.datasets import mnist
from MLP_active_functions import Active_functions

class MLP:

    def mnist(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        return train_X, train_y, test_X, test_y

    def input_layer(self):
        train_X, train_y, test_X, test_y = self.mnist()
        validate_X = train_X[100:200]
        validate_y = train_y[100:200]
        return train_X[:100], train_y[:100], test_X[:10], test_y[:10], validate_X, validate_y
        
    def hidden_layer(self, x, number, weights_for_all_neurons, bias_for_all_neurons, fun, which_layer):
        af = Active_functions()
        sum_for_all = []
        a_for_all_neurons = []
        for c in range(len(x)): #dla kazdego obrazka
            a=[]
            sums=[]
            for b in range(number): #dla kazdego neuronu
                sum= self.sum_all(x[c], weights_for_all_neurons[b][c], bias_for_all_neurons[b], which_layer)
                sums.append(sum)
                a.append(af.choose_fun(fun, sum))
            a_for_all_neurons.append(a) 
            sum_for_all.append(sums)
        return a_for_all_neurons, sum_for_all

    def output_layer(self, x, number, weights_for_all_neurons, bias_for_all_neurons):
        af = Active_functions()
        a_for_all_neurons = []
        for c in range(len(x)): #dla kazdego obrazka
            a=[]
            for b in range(number): #dla kazdego neuronu
                sum = self.sum_all(x[c], weights_for_all_neurons[b][c], bias_for_all_neurons[b],2)
                a.append(sum)
            a_for_all_neurons.append(a)

        y=[]
        for i in range(len(a_for_all_neurons)): y.append(af.softmax(a_for_all_neurons[i]))

        return y, a_for_all_neurons

    def diff_hiden(self,weights,diffs,bias,fun,sums):
        diffs_h=[]
        af = Active_functions()
        for i in range (len(sums)):
            for j in range(len(sums[i])):
                z = af.choose_deriv(sums[i][j],fun)
                sum=(diffs[i] * weights[j][i]) + bias[j]
            diffs_h.append(z*sum)
        return diffs_h

    def diff_output(self,y_predict, labels, sums):
        af = Active_functions()
        diffs=[]
        for i in range(len(y_predict)):
            diff=labels[i] - max(y_predict[i])
            diff_soft = diff * af.softmax_max(sums[i])
            diffs.append(diff_soft)
        return diffs

    def weights_update(self,weights,mi,a,diff):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] += mi * a[j][i] * diff[j]
        return weights

    def bias_update(self,bias,mi,diff):
        for i in range(len(bias)): bias[i] += mi * (sum(diff)/len(diff))
        return bias

    def random_weights_and_bias(self, number, x, whmax, whmin):
        weights_for_all_neurons =[]
        bias_for_all_neurons=self.random_bias(number)
        for _ in range (number):
            weights_for_neuron = []
            for _ in range (x):
                weights_for_neuron.append(random.uniform(whmin, whmax))
            weights_for_all_neurons.append(weights_for_neuron)
            
        return weights_for_all_neurons, bias_for_all_neurons

    def random_bias(self,number):
        bias_for_all_neurons=[]
        for _ in range(number):
            bias_for_all_neurons.append(random.uniform(-0.2, 0.2))
        return bias_for_all_neurons

    def set_weights_and_bias(self,number,whmin,whmax,X,WEIGHTS,BIAS):
        WFHL, BFHL =self.random_weights_and_bias(number,X,whmax, whmin)
        WSHL, BSHL =self.random_weights_and_bias((int)((number*0.7)),X,whmax, whmin)
        WOHL, BOHL =self.random_weights_and_bias(10,X,whmax, whmin)
        WEIGHTS.append(WFHL)
        WEIGHTS.append(WSHL)
        WEIGHTS.append(WOHL)
        BIAS.append(BFHL)
        BIAS.append(BSHL)
        BIAS.append(BOHL)
        return WEIGHTS,BIAS

    def sum_all(self, X, w, b, l):
        sum = 0
        if(l==1):
            for i in range(len(X)):
                for j in range(len(X[i])):
                    sum += (round(X[i][j],2) * round(w,2)) + b
        else:
            for j in range(len(X)): sum +=(X[j] * w) + b
        return sum

