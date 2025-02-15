# Iris Neural Network

This folder contains a simple neural network implemented from scratch in C to classify flowers from the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris). The neural network is trained using backpropagation with gradient descent.

## The Iris Dataset

The Iris dataset is a well-known dataset in machine learning, consisting of 150 samples of iris flowers classified into three species:
- Iris Setosa (Label: 0)
- Iris Versicolor (Label: 1)
- Iris Virginica (Label: 2)

Each sample has four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Network Architecture
- Input Layer: 4 neurons (one for each feature)
- Hidden Layer: 8 neurons (configurable)
- Output Layer: 3 neurons (one for each class, using softmax activation)
- Activation Function: tanh in the hidden layer, softmax in the output layer
- Loss Function: Categorical Cross-Entropy
- Training Method: Backpropagation with Gradient Descent

## Expected Output

After training, the network should achieve around 95-100% accuracy on the test set.
