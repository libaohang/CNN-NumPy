
# Convolutional Neural Network with Pure Numpy Implementation
<br>

## Description
A convolutional neural network with all layers implemented using only Numpy<br>
<br>

## Classifying MNIST and CIFAR-10 Datasets
Using the layers I wrote in Numpy, I built 3 different networks of increasing complexity. I used the first 2 networks to classify MNIST, and the 3rd network to classify CIFAR-10<br>
For context, MNIST is a dataset of grayscale images of handwritten numbers 0 to 9, and CIFAR is a dataset of colored images of 10 types of objects, such as birds, planes, trucks, etc<br>
The details on each network and their performance on the datasets are described below: <br>
Note: errors are calculated using cross-entropy loss<br>

### Network 1
Network 1 has a total of 1 convolution layer and 4 filters: Convo(4) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 1 on MNIST:**
<img width="1291" height="795" alt="image" src="https://github.com/user-attachments/assets/c072ca76-d054-400b-af7c-7528c5f2fd45" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 25 epochs<br>
<br>
Network 1 achieves a test error of 0.14, which is equivalent to **96% accuracy** on MNIST. This is a good result for a CNN with 1 convolution layer and 4 filters. <br>

<br>

### Network 2
Network 2 has a total of 2 convolution layers and 32 filters: Convo(16) -> ReLu -> MaxPool -> Convo(16) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 2 on MNIST:**
<img width="1296" height="815" alt="image" src="https://github.com/user-attachments/assets/42433137-5e8e-4c4b-86fb-78fb519518c6" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 30 epochs<br>
<br>
Network 2 achieves a test error of 0.07, which is equivalent to **97.8% accuracy** on MNIST. The learning curve is noticeably steeper with 2 convolution layers and more filters. With even more convolution layers, I would be able to reach 99% or more on MNIST, but I decided to move on to classifying CIFAR-10<br>

<br>

### Network 3
Network 3 has a total of 3 convolution layers and 56 filters: Convo(8) -> ReLu -> MaxPool -> Convo(16) -> ReLu -> Convo(32) -> ReLu -> MaxPool -> Flatten -> Dense -> ReLu -> Dense -> SoftMax<br>
**Error over Epochs Trained for network 3 on CIFAR-10:**
<img width="1130" height="802" alt="image" src="https://github.com/user-attachments/assets/606dce1a-53a2-4dd4-889e-dab92258621d" />
__Key:__ <br>
Green line: error on each epoch<br>
Red line: final test error after 25 epochs<br>
<br>
Network 3 achieves a test error of 1.16, which is equivalent to **60% accuracy** on CIFAR-10. This is actually a very good accuracy for a basic network like this without batch normalization or data augmentation.
The change from grayscale numbers in MNIST to colored objects of CIFAR-10 made learning much more difficult, as seen in how the learning curve is much flatter than the previous classifications. 
I planned to train a 4th network with even more filters and convolution layers to classify CIFAR-10, but it takes too long to run because the Numpy implementation does not support GPU acceleration. 
However, I am sure the 4th network has the potential to reach 80% accuracy with additional improvements, such as batch normalization layers.
