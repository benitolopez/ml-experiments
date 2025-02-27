/**
 * neural_network.h
 *
 * Header for neural network implementation for MNIST digit classification.
 * Provides a simple feedforward neural network with backpropagation training.
 */
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

// Network architecture constants
#define INPUTS_SIZE 784 // 28x28 pixels
#define OUTPUTS_SIZE 10 // 10 possible digits (0-9)

/**
 * Layer structure representing a single layer in the neural network.
 */
struct Layer {
  int size;       // Number of neurons in this layer
  float *values;  // Neuron activation values
  float *weights; // Weights connecting to next layer (size * next_layer_size)
  float *biases;  // Biases for next layer (next_layer_size)
};

/**
 * Network structure containing all layers of the neural network.
 */
struct Network {
  int num_layers;       // Total number of layers including input and output
  struct Layer *layers; // Array of layers
};

/**
 * Create a new neural network with specified architecture.
 *
 * @param layer_sizes Array containing the size of each layer
 * @param num_layers  Number of layers in the network
 * @return            Pointer to the created network or NULL on failure
 */
struct Network *create_network(int *layer_sizes, int num_layers);

/**
 * Free memory allocated for the network.
 *
 * @param net Pointer to the network to free
 */
void free_network(struct Network *net);

/**
 * Perform forward pass (inference) through the network.
 *
 * @param net   Pointer to the network
 * @param input Input data (must match input layer size)
 */
void forward_pass(struct Network *net, float *input);

/**
 * Perform backward pass (backpropagation) for training.
 *
 * @param net           Pointer to the network
 * @param input         Input data
 * @param true_label    Correct output label (ground truth)
 * @param learning_rate Learning rate for gradient descent
 */
void backward_pass(struct Network *net, float *input, int true_label,
                   float learning_rate);

/**
 * Train the network on a dataset.
 *
 * @param net               Pointer to the network
 * @param train_images      Training images
 * @param train_labels      Training labels
 * @param num_train_samples Number of training samples
 * @param learning_rate     Learning rate for gradient descent
 * @param num_epochs        Number of training epochs
 * @param batch_size        Mini-batch size for stochastic gradient descent
 */
void train_network(struct Network *net, float **train_images, int *train_labels,
                   int num_train_samples, float learning_rate, int num_epochs,
                   int batch_size);

/**
 * Shuffle indices for randomized training.
 *
 * @param indices Array of indices to shuffle
 * @param count   Number of indices
 */
void shuffle_indices(int *indices, int count);

/**
 * Save network weights and biases to a file.
 *
 * @param net      Pointer to the network
 * @param filename Output filename
 * @return         0 on success, 1 on failure
 */
int save_network(struct Network *net, const char *filename);

/**
 * Load network weights and biases from a file.
 *
 * @param net      Pointer to the initialized network
 * @param filename Input filename
 * @return         0 on success, 1 on failure
 */
int load_network(struct Network *net, const char *filename);

/**
 * Evaluate network performance on test data.
 *
 * @param net             Pointer to the network
 * @param test_images     Test images
 * @param test_labels     Test labels
 * @param num_test_samples Number of test samples
 */
void evaluate_network(struct Network *net, float **test_images,
                      int *test_labels, int num_test_samples);

/**
 * ReLU activation function: f(x) = max(0, x)
 *
 * @param x Input value
 * @return  Activated value
 */
float relu(float x);

/**
 * Softmax activation function for output layer.
 * Converts raw outputs to probability distribution.
 *
 * @param input Array of input values
 * @param size  Size of the input array
 */
void softmax(float *input, int size);

/**
 * Calculate cross-entropy loss between network output and true label.
 *
 * @param network_output Softmax output from the network
 * @param true_label     Ground truth label
 * @return               Loss value (lower is better)
 */
float cross_entropy_loss(float *network_output, int true_label);

#endif // NEURAL_NETWORK_H
