from Layers import MaxPoolingLayer, ConvolutionLayer, DenseLayer, FlattenLayer
from Activations import ReLu, SoftMax
from CNN import trainCNN, testCNN
import numpy as np
from tensorflow.keras.datasets import mnist
from TestingTools import DigitGUI

def cross_entropy(true, pred):
    if true.ndim == 2:          # one-hot
        true = np.argmax(true, axis=1)
    return -np.log(pred[np.arange(len(true)), true] + 1e-10)

def cross_entropy_prime(true, pred):
    # Convert one-hot → integer labels
    if true.ndim == 2 and true.shape[1] > 1:
        true = np.argmax(true, axis=1)
    y = np.zeros_like(pred)
    y[np.arange(len(pred)), true] = 1
    return (pred - y) / len(pred)

# Networks for MNIST:

# network1 reach 95.78% test accuracy after 25 epochs
network1 = [                              # 28 x 28 x 1
    ConvolutionLayer(3, 4, 1, 0.1, 0.9),  # 28 x 28 x 4
    ReLu(),
    MaxPoolingLayer(2),                   # 14 x 14 x 4
    FlattenLayer(),
    DenseLayer(784, 128, 0.1, 0.9),     
    ReLu(),
    DenseLayer(128, 10, 0.1, 0.9),
    SoftMax()
]

# network2 reach 97.3% test accuracy after 25 epochs
network2 = [                               # 28 x 28 x 1
    ConvolutionLayer(3, 16, 1, 0.1, 0.9),  # 28 x 28 x 16
    ReLu(),
    MaxPoolingLayer(2),                    # 14 x 14 x 16
    ConvolutionLayer(3, 16, 16, 0.1, 0.9), # 14 x 14 x 16
    ReLu(),
    MaxPoolingLayer(2),                    # 7 x 7 x 16
    FlattenLayer(),
    DenseLayer(784, 200 , 0.05, 0.9),
    ReLu(),
    DenseLayer(200, 10, 0.05, 0.9),
    SoftMax()

]

def classifyMNIST(network):

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = xTrain.astype(np.float32) / 255.0
    xTest  = xTest.astype(np.float32) / 255.0

    xTrain = xTrain[:, :, :, None]
    xTest = xTest[:, :, :, None]

    trainedNetwork = trainCNN(network, cross_entropy, cross_entropy_prime, xTrain, yTrain, 25, 40)

    testCNN(trainedNetwork, cross_entropy, xTest, yTest)

    gui = DigitGUI(trainedNetwork)
    gui.run()

# Networks for CIFAR-10

network3 = [                               # 32 x 32 x 3
    ConvolutionLayer(3, 16, 3, 0.1, 0.9),  # 32 x 32 x 16
    ReLu(),
    MaxPoolingLayer(2),                    # 16 x 16 x 16
    ConvolutionLayer(3, 32, 16, 0.1, 0.9), # 16 x 16 x 32
    ReLu(),
    MaxPoolingLayer(2),                    # 8 x 8 x 32
    ConvolutionLayer(3, 16, 32, 0.1, 0.9), # 8 x 8 x 16
    ReLu(),
    FlattenLayer(),
    DenseLayer(1024, 250 , 0.05, 0.9),
    ReLu(),
    DenseLayer(250, 10, 0.05, 0.9),
    SoftMax()

]

def classifyCIFAR10(network):
    from tensorflow.keras.datasets import cifar10

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    yTrain_onehot = np.eye(10)[yTrain.reshape(-1)]
    yTest_onehot  = np.eye(10)[yTest.reshape(-1)]

    xTrain = xTrain.astype("float32") / 255.0
    xTest = xTest.astype("float32") / 255.0

    trainedNetwork = trainCNN(network, cross_entropy, cross_entropy_prime, xTrain, yTrain_onehot, 20, 40)

    testCNN(trainedNetwork, cross_entropy, xTest, yTest_onehot)


if __name__ == '__main__':
    #classifyMNIST(network2)
    classifyCIFAR10(network3)
