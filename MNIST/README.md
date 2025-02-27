# MNIST Digit Recognition

This folder contains a neural network implemented from scratch in C to classify handwritten digits from the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). The neural network is trained using backpropagation with stochastic gradient descent and mini-batch optimization.

## The MNIST Dataset

The MNIST dataset is a widely-used benchmark in machine learning, consisting of 70,000 grayscale images of handwritten digits (0-9):
- 60,000 training images
- 10,000 test images

Each image is 28x28 pixels, with pixel values ranging from 0 to 255, normalized to [0,1] for training.

## Network Architecture

- Input Layer: 784 neurons (28x28 pixels flattened)
- Hidden Layer(s): Configurable (default: 128 neurons)
- Output Layer: 10 neurons (one for each digit, using softmax activation)
- Activation Function: ReLU in hidden layers, softmax in output layer
- Loss Function: Cross-Entropy
- Training Method: Mini-batch Stochastic Gradient Descent

## Implementation Features

- Modular C implementation with separate components:
  - Neural network module with forward/backward propagation
  - MNIST binary file loader
  - Training and evaluation functions
- Support for saving and loading trained weights
- Support for configurable network architectures (variable hidden layers)

## Expected Performance

After training, the network typically achieves:
- ~97-98% accuracy on the test set with a single hidden layer

## Usage

```bash
# Compile the project
make

# Train the network (if no saved weights exist)
./build/mnist_nn
```
