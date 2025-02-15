# XOR Neural Network

This folder contains a simple neural network implemented from scratch in C to solve the XOR problem. The neural network is trained using backpropagation with gradient descent.

## The XOR Problem

The XOR (exclusive OR) function is a classic non-linear problem, defined as:

| Input A  | Input B | XOR Output |
| -------- | ------- | ---------- |
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

## Network Architecture
-	Input Layer: 2 neurons (configurable)
-	Hidden Layer: 4 neurons (configurable)
-	Output Layer: 1 neuron (with sigmoid activation)
-	Activation Function: tanh, used instead of ReLU because ReLU caused issues in training
-	Loss Function: Binary Cross-Entropy

## Expected Output

After training, the network should correctly approximate XOR outputs:

```
Network Evaluation:
Input: [0, 0], Target: 0, Prediction: ~0.00
Input: [0, 1], Target: 1, Prediction: ~1.00
Input: [1, 0], Target: 1, Prediction: ~1.00
Input: [1, 1], Target: 0, Prediction: ~0.00
```
