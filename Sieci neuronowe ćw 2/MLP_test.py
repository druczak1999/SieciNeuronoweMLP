from MLP_alg import MLP_alogorythm

if __name__ == '__main__':
    mlp = MLP_alogorythm()
    # epochs=mlp.learn_algorythm_batch(30, -0.2, 0.2, 0.2, 0.1, 100,1000)
    for i in range(3,5):
        print(10*i+5)
        epochs = mlp.learn_algorythm(10*i+5,-0.2,0.2,0.1,0.1)
    # epochs = mlp.learn_algorythm_early_stopping(50,-0.2,0.2,0.1,0.1,0.5)
           