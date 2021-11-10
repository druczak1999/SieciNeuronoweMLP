import random
INFINITY = float('inf')
WEIGHTS =[]
BIAS = []
from MLP import MLP

class MLP_alogorythm:
   
    def algorythm(self,X,Y,number,WEIGHTS,BIAS,fun1,fun2, epochs, prog, mi):
        mlp = MLP()
        print("layers")
        a1, sum_for_all_1 = mlp.hidden_layer(X,number,WEIGHTS[0],BIAS[0],fun1, 1)
        a2, sum_for_all = mlp.hidden_layer(a1,(int)((number*0.7)),WEIGHTS[1],BIAS[1], fun2, 2)
        y, a_for_all_neurons = mlp.output_layer(a1,10,WEIGHTS[2],BIAS[2])  
        print("differences")
        diffs = mlp.diff_output(y,Y,a_for_all_neurons)
        diffs_h2 = mlp.diff_hiden(WEIGHTS[1],diffs,BIAS[1],fun2,sum_for_all)
        diffs_h1 = mlp.diff_hiden(WEIGHTS[0],diffs,BIAS[0],fun1,sum_for_all_1)
        print("upadate weights")
        WEIGHTS[2] = mlp.weights_update(WEIGHTS[2],mi,y,diffs)
        WEIGHTS[1] = mlp.weights_update(WEIGHTS[1],mi,a2,diffs_h2)
        WEIGHTS[0] = mlp.weights_update(WEIGHTS[0],mi,a1,diffs_h1)
        print("upadate biases")
        BIAS[2] = mlp.bias_update(BIAS[2], mi, diffs)
        BIAS[1] = mlp.bias_update(BIAS[1], mi, diffs_h2)
        BIAS[0] = mlp.bias_update(BIAS[0], mi, diffs_h1)
        diffs_pow=[]
        for i in range(len(diffs)):
            diffs_pow.append(diffs[i]**2)    
        prog = sum(diffs_pow)/2
        epochs+=1
        print("PROG",prog)
        return WEIGHTS, BIAS, epochs, prog

    def learn_algorythm(self, number, whmin, whmax, mi, threshold):
        mlp = MLP()
        X, Y, x_test, y_test, v_X, v_y = mlp.input_layer()
        WEIGHTS=[]
        BIAS=[]
        print("weights")
        WEIGHTS,BIAS = mlp.set_weights_and_bias(number, whmin, whmax, len(X), WEIGHTS, BIAS)
        prog = INFINITY
        epochs = 0
        fun1 = "hiperbola"
        fun2 = "sigma"
        while prog > threshold: WEIGHTS, BIAS, epochs, prog = self.algorythm(X, Y, number, WEIGHTS, BIAS, fun1, fun2, epochs, prog, mi)
        print(epochs)
        return epochs
        
    def learn_algorythm_early_stopping(self, number, whmin, whmax, mi, threshold, validate_threshold):
        mlp = MLP()
        X, Y, x_test, y_test, v_X, v_y = mlp.input_layer()
        WEIGHTS=[]
        BIAS=[]
        print("weights")
        WEIGHTS,BIAS = mlp.set_weights_and_bias(number,whmin, whmax,len(X),WEIGHTS,BIAS)
        prog = INFINITY
        prog_v = 0
        epochs = 0
        fun1 = "relu"
        fun2 = "hiperbola"
        while prog > threshold and prog_v<validate_threshold:
            WEIGHTS,BIAS,epochs,prog = self.algorythm(X, Y, number, WEIGHTS, BIAS, fun1, fun2, epochs, prog, mi)
            prog_v = self.validate_threshold(v_X, number, fun1, fun2, v_y, prog_v, WEIGHTS, BIAS)
        print(epochs)
        return epochs
        
    def validate_threshold(self, v_X, number, fun1, fun2, v_y, prog_v, WEIGHTS, BIAS): 
        mlp = MLP()
        print("layers")
        a1_v, sum_for_all_1_v = mlp.hidden_layer(v_X, number, WEIGHTS[0], BIAS[0], fun1, 1)
        a2_v, sum_for_all_v = mlp.hidden_layer(a1_v, (int)((number*0.7)), WEIGHTS[1], BIAS[1], fun2, 2)
        y_v, a_for_all_neurons_v = mlp.output_layer(a2_v, 10, WEIGHTS[2], BIAS[2])  
        print("differences")
        diffs_v = mlp.diff_output(y_v, v_y, a_for_all_neurons_v)
        
        diffs_pow_v=[]
        for i in range(len(diffs_v)):
            diffs_pow_v.append(diffs_v[i]**2)    
        new_prog_v = sum(diffs_pow_v)/2
        prog_v = new_prog_v - prog_v
        print("Val",prog_v)
        return prog_v

    # for batch

    def input_layer_batch(self,batch,i,train_X, train_y):
        return train_X[i*batch:batch+i*batch], train_y[i*batch:batch+i*batch]

    def read_mnist_batch(self,len):
        mlp = MLP()
        train_X, train_y, test_X, test_y = mlp.mnist()
        return train_X, train_y, test_X, test_y

    def learn_algorythm_batch(self, number, whmin, whmax, mi, threshold, batch, len_of_input):
        mlp = MLP()
        prog=INFINITY
        WEIGHTS=[]
        BIAS=[]
        prog = INFINITY
        epochs = 0
        ep=0
        fun1 = "sigma"
        fun2 = "hiperbola"
        WEIGHTS,BIAS = mlp.set_weights_and_bias(number, whmin, whmax, batch, WEIGHTS, BIAS)
        while prog > threshold:
            train_X, train_y, test_X, test_y = mlp.mnist();
            rng_state = random.getstate()
            random.shuffle(train_X)
            random.setstate(rng_state)
            random.shuffle(train_y)
            progs=[]
            for i in range(int(len_of_input/batch)):
                X, Y = self.input_layer_batch(batch,i, train_X, train_y)
                print("weights")
                WEIGHTS,BIAS,epochs,prog1 = self.algorythm(X, Y, number, WEIGHTS, BIAS, fun1, fun2, epochs, prog, mi)
                progs.append(prog1)
            prog = sum(progs)/len(progs)
            print("PROG PO CALYM",prog)
            ep+=1
        print(epochs)
        return ep