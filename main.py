from Layers import MaxPoolingLayer, ConvolutionLayer, DenseLayer, FlattenLayer
from Activations import ReLu, SoftMax
from CNN import trainCNN, testCNN
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

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

# network1 reach 96.05% test accuracy on MNIST after 25 epochs
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

# network2 reach 97.78% test accuracy on MNIST after 30 epochs
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


# Networks for CIFAR-10

# network3 reach 60.04% test accuracy on CIFAR-10 after 25 epochs
network3 = [                                  # 32 x 32 x 3
    ConvolutionLayer(3, 8, 3, 0.1, 0.9),      # 32 x 32 x 8
    ReLu(),
    MaxPoolingLayer(2),                       # 16 x 16 x 8
    ConvolutionLayer(3, 16, 8, 0.1, 0.9),     # 16 x 16 x 16
    ReLu(),
    ConvolutionLayer(3, 32, 16, 0.1, 0.9),    # 16 x 16 x 32
    ReLu(),
    MaxPoolingLayer(2),                       # 8 x 8 x 32

    FlattenLayer(),              
    DenseLayer(2048, 256, 0.1, 0.9),
    ReLu(),
    DenseLayer(256, 10, 0.1, 0.9),
    SoftMax()
]

network4 = [                                  # 32 x 32 x 3
    ConvolutionLayer(3, 32, 3, 0.1, 0.9),     # 32 x 32 x 32
    ReLu(),
    ConvolutionLayer(3, 32, 32, 0.1, 0.9),    # 32 x 32 x 32
    ReLu(),
    MaxPoolingLayer(2),                       # 16 x 16 x 32

    ConvolutionLayer(3, 64, 32, 0.1, 0.9),    # 16 x 16 x 64
    ReLu(),
    ConvolutionLayer(3, 64, 64, 0.1, 0.9),    # 16 x 16 x 64
    ReLu(),
    MaxPoolingLayer(2),                       # 8 x 8 x 64

    FlattenLayer(),                           # 4096 units
    DenseLayer(4096, 256, 0.05, 0.9),
    ReLu(),
    DenseLayer(256, 10, 0.05, 0.9),
    SoftMax()
]

def classifyCIFAR10(network):
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    yTrain_onehot = np.eye(10)[yTrain.reshape(-1)]
    yTest_onehot  = np.eye(10)[yTest.reshape(-1)]

    xTrain = xTrain.astype("float32") / 255.0
    xTest = xTest.astype("float32") / 255.0

    trainedNetwork = trainCNN(network, cross_entropy, cross_entropy_prime, xTrain, yTrain_onehot, 25, 40)

    testCNN(trainedNetwork, cross_entropy, xTest, yTest_onehot)


if __name__ == '__main__':
    classifyMNIST(network1)
    #classifyCIFAR10(network3)
