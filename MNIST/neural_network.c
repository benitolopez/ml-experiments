/**
 * neural_network.c
 *
 * Implementation of a feedforward neural network with backpropagation.
 * Includes activation functions, forward/backward propagation algorithms,
 * and utility functions for training, evaluation, and model persistence.
 */
#include "neural_network.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * ReLU activation function: f(x) = max(0, x)
 * Used for hidden layers to introduce non-linearity.
 *
 * @param x Input value
 * @return  max(0, x)
 */
float relu(float x) { return x > 0 ? x : 0; }

/**
 * Softmax activation function for output layer.
 * Converts raw outputs to probability distribution.
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 *
 * @param input Array of input values
 * @param size  Size of the input array
 */
void softmax(float *input, int size) {
  // Find max value for numerical stability (prevents overflow)
  float max = input[0];
  float sum = 0.0;

  for (int i = 1; i < size; i++) {
    if (input[i] > max)
      max = input[i];
  }

  // Compute exp(x_i - max) and sum
  for (int i = 0; i < size; i++) {
    input[i] = exp(input[i] - max);
    sum += input[i];
  }

  // Normalize to get probabilities
  for (int i = 0; i < size; i++) {
    input[i] /= sum;
  }
}

/**
 * Calculate cross-entropy loss between network output and true label.
 * Formula: loss = -log(output[true_label])
 *
 * @param network_output Softmax output from the network
 * @param true_label     Ground truth label
 * @return               Loss value (lower is better)
 */
float cross_entropy_loss(float *network_output, int true_label) {
  // Small epsilon value to avoid log(0)
  const float epsilon = 1e-15;

  // Get probability for the true class, with epsilon to avoid log(0)
  float true_prob = network_output[true_label] + epsilon;

  // Return negative log probability
  return -log(true_prob);
}

/**
 * Free memory allocated for the network.
 *
 * @param net Pointer to the network to free
 */
void free_network(struct Network *net) {
  for (int i = 0; i < net->num_layers; i++) {
    if (i < net->num_layers - 1) {
      free(net->layers[i].biases);
      free(net->layers[i].weights);
    }

    free(net->layers[i].values);
  }

  free(net->layers);
  free(net);
}

/**
 * Clean up partially allocated network when creation fails.
 *
 * @param net           Pointer to the network
 * @param current_layer Current layer during allocation
 */
void cleanup_partial_network(struct Network *net, int current_layer) {
  // Free all layers up to current_layer
  for (int i = 0; i < current_layer; i++) {
    free(net->layers[i].values);
    if (i < net->num_layers - 1) {
      free(net->layers[i].weights);
      free(net->layers[i].biases);
    }
  }
  free(net->layers);
  free(net);
}

/**
 * Save network weights and biases to a binary file.
 *
 * @param net      Pointer to the network
 * @param filename Output filename
 * @return         0 on success, 1 on failure
 */
int save_network(struct Network *net, const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    printf("Error: Could not open file %s for writing\n", filename);
    return 1;
  }

  // Write network structure (number of layers and layer sizes)
  fwrite(&net->num_layers, sizeof(int), 1, file);

  for (int i = 0; i < net->num_layers; i++) {
    fwrite(&net->layers[i].size, sizeof(int), 1, file);
  }

  // Write weights and biases
  for (int i = 0; i < net->num_layers - 1; i++) {
    int weights_count = net->layers[i].size * net->layers[i + 1].size;
    fwrite(net->layers[i].weights, sizeof(float), weights_count, file);
    fwrite(net->layers[i].biases, sizeof(float), net->layers[i + 1].size, file);
  }

  fclose(file);
  printf("Network saved to %s\n", filename);
  return 0;
}

/**
 * Load network weights and biases from a binary file.
 *
 * @param net      Pointer to the initialized network
 * @param filename Input filename
 * @return         0 on success, 1 on failure
 */
int load_network(struct Network *net, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    printf("Error: Could not open file %s for reading\n", filename);
    return 1;
  }

  // Read and verify network structure
  int num_layers;
  fread(&num_layers, sizeof(int), 1, file);

  if (num_layers != net->num_layers) {
    printf("Error: Network structure mismatch (expected %d layers, found %d)\n",
           net->num_layers, num_layers);
    fclose(file);
    return 1;
  }

  // Verify layer sizes
  for (int i = 0; i < num_layers; i++) {
    int layer_size;
    fread(&layer_size, sizeof(int), 1, file);

    if (layer_size != net->layers[i].size) {
      printf("Error: Layer size mismatch at layer %d (expected %d, found %d)\n",
             i, net->layers[i].size, layer_size);
      fclose(file);
      return 1;
    }
  }

  // Read weights and biases
  for (int i = 0; i < net->num_layers - 1; i++) {
    int weights_count = net->layers[i].size * net->layers[i + 1].size;
    fread(net->layers[i].weights, sizeof(float), weights_count, file);
    fread(net->layers[i].biases, sizeof(float), net->layers[i + 1].size, file);
  }

  fclose(file);
  printf("Network loaded from %s\n", filename);
  return 0;
}

/**
 * Evaluate network on test data.
 * Calculates and reports loss and accuracy metrics.
 *
 * @param net             Pointer to the network
 * @param test_images     Test images
 * @param test_labels     Test labels
 * @param num_test_samples Number of test samples
 */
void evaluate_network(struct Network *net, float **test_images,
                      int *test_labels, int num_test_samples) {
  int correct_predictions = 0;
  float total_loss = 0.0;

  printf("Evaluating network on %d test samples...\n", num_test_samples);

  for (int i = 0; i < num_test_samples; i++) {
    // Perform inference
    forward_pass(net, test_images[i]);

    // Calculate loss
    float loss = cross_entropy_loss(net->layers[net->num_layers - 1].values,
                                    test_labels[i]);
    total_loss += loss;

    // Find predicted digit (highest probability)
    int predicted_digit = 0;
    float max_prob = net->layers[net->num_layers - 1].values[0];

    for (int j = 1; j < OUTPUTS_SIZE; j++) {
      if (net->layers[net->num_layers - 1].values[j] > max_prob) {
        max_prob = net->layers[net->num_layers - 1].values[j];
        predicted_digit = j;
      }
    }

    // Count correct predictions
    if (predicted_digit == test_labels[i]) {
      correct_predictions++;
    }

    // Display progress periodically
    if ((i + 1) % 1000 == 0 || i == num_test_samples - 1) {
      printf("Processed %d/%d test samples\r", i + 1, num_test_samples);
      fflush(stdout);
    }
  }

  // Calculate and print metrics
  float accuracy = 100.0f * correct_predictions / num_test_samples;
  float avg_loss = total_loss / num_test_samples;

  printf("\nTest results - Loss: %.4f - Accuracy: %.2f%% (%d/%d correct)\n",
         avg_loss, accuracy, correct_predictions, num_test_samples);
}

/**
 * Create a new neural network with specified architecture.
 * Initializes weights with small random values and biases with zeros.
 *
 * @param layer_sizes Array containing the size of each layer
 * @param num_layers  Number of layers in the network
 * @return            Pointer to the created network or NULL on failure
 */
struct Network *create_network(int *layer_sizes, int num_layers) {
  // Seed random number generator for weight initialization
  srand(time(NULL));

  // Allocate memory for network structure
  struct Network *net = malloc(sizeof(struct Network));
  if (net == NULL) {
    printf("Error allocating network\n");
    return NULL;
  }

  net->num_layers = num_layers;
  net->layers = malloc(num_layers * sizeof(struct Layer));
  if (net->layers == NULL) {
    free(net);
    printf("Error allocating layers\n");
    return NULL;
  }

  // Initialize each layer
  for (int i = 0; i < num_layers; i++) {
    net->layers[i].size = layer_sizes[i];

    // Allocate memory for neuron values
    net->layers[i].values = malloc(layer_sizes[i] * sizeof(float));
    if (net->layers[i].values == NULL) {
      printf("Error allocating layer values\n");
      cleanup_partial_network(net, i);
      return NULL;
    }

    // For all except the output layer, allocate weights and biases
    if (i < num_layers - 1) {
      // Allocate memory for weights: current_layer_size * next_layer_size
      net->layers[i].weights =
          malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));

      if (net->layers[i].weights == NULL) {
        printf("Error allocating layer weights\n");
        cleanup_partial_network(net, i);
        return NULL;
      }

      // Initialize weights with small random values (He initialization)
      for (int j = 0; j < layer_sizes[i]; j++) {
        for (int k = 0; k < layer_sizes[i + 1]; k++) {
          int weight_index = j * layer_sizes[i + 1] + k;

          // Small random values between -0.1 and 0.1
          net->layers[i].weights[weight_index] =
              ((float)rand() / RAND_MAX - 0.5) * 0.2;
        }
      }

      // Allocate and initialize biases to zero
      net->layers[i].biases = malloc(layer_sizes[i + 1] * sizeof(float));
      if (net->layers[i].biases == NULL) {
        printf("Error allocating layer biases\n");
        cleanup_partial_network(net, i);
        return NULL;
      }

      for (int j = 0; j < layer_sizes[i + 1]; j++) {
        net->layers[i].biases[j] = 0.0;
      }
    } else {
      // Output layer doesn't need weights/biases (they belong to previous
      // layer)
      net->layers[i].weights = NULL;
      net->layers[i].biases = NULL;
    }
  }

  return net;
}

/**
 * Perform forward pass (inference) through the network.
 *
 * @param net   Pointer to the network
 * @param input Input data (must match input layer size)
 */
void forward_pass(struct Network *net, float *input) {
  // Copy input to first layer's values
  for (int i = 0; i < net->layers[0].size; i++) {
    net->layers[0].values[i] = input[i];
  }

  // Process each layer
  for (int layer = 0; layer < net->num_layers - 1; layer++) {
    // For each neuron in next layer
    for (int j = 0; j < net->layers[layer + 1].size; j++) {
      float sum = 0.0;

      // Compute weighted sum of inputs
      for (int i = 0; i < net->layers[layer].size; i++) {
        int weight_index = i * net->layers[layer + 1].size + j;
        sum += net->layers[layer].values[i] *
               net->layers[layer].weights[weight_index];
      }

      // Add bias term
      sum += net->layers[layer].biases[j];

      // Apply activation function
      if (layer < net->num_layers - 2) {
        // ReLU for hidden layers
        net->layers[layer + 1].values[j] = relu(sum);
      } else {
        // Linear output for last layer (before softmax)
        net->layers[layer + 1].values[j] = sum;
      }
    }

    // Apply softmax to output layer
    if (layer == net->num_layers - 2) {
      softmax(net->layers[layer + 1].values, net->layers[layer + 1].size);
    }
  }
}

/**
 * Perform backward pass (backpropagation) for training.
 *
 * @param net           Pointer to the network
 * @param input         Input data
 * @param true_label    Correct output label (ground truth)
 * @param learning_rate Learning rate for gradient descent
 */
void backward_pass(struct Network *net, float *input, int true_label,
                   float learning_rate) {
  // Forward pass to get all activations
  forward_pass(net, input);

  // Get output layer index
  int output_layer_idx = net->num_layers - 1;
  int hidden_layer_idx = output_layer_idx - 1;

  // For softmax with cross-entropy, output error is (predicted - actual)
  // Create error arrays for each layer
  float *output_error =
      malloc(net->layers[output_layer_idx].size * sizeof(float));
  if (!output_error) {
    printf("Error: Failed to allocate memory for output errors\n");
    return;
  }

  // Calculate output layer error
  for (int i = 0; i < net->layers[output_layer_idx].size; i++) {
    output_error[i] =
        net->layers[output_layer_idx].values[i] - (i == true_label ? 1.0 : 0.0);
  }

  // Allocate memory for hidden layer errors (for each layer before output)
  float **hidden_errors = malloc((output_layer_idx - 1) * sizeof(float *));
  if (!hidden_errors) {
    printf("Error: Failed to allocate memory for hidden errors\n");
    free(output_error);
    return;
  }

  // Allocate error arrays for each hidden layer
  for (int layer = 0; layer < output_layer_idx - 1; layer++) {
    hidden_errors[layer] = calloc(net->layers[layer + 1].size, sizeof(float));
    if (!hidden_errors[layer]) {
      printf("Error: Failed to allocate memory for hidden layer %d errors\n",
             layer);
      // Clean up previously allocated arrays
      for (int j = 0; j < layer; j++) {
        free(hidden_errors[j]);
      }
      free(hidden_errors);
      free(output_error);
      return;
    }
  }

  // Backpropagate error from output to hidden layers
  // Start with last hidden layer
  for (int i = 0; i < net->layers[hidden_layer_idx].size; i++) {
    float error_sum = 0.0;

    // Sum contributions to error from each output neuron
    for (int j = 0; j < net->layers[output_layer_idx].size; j++) {
      int weight_index = i * net->layers[output_layer_idx].size + j;
      error_sum +=
          output_error[j] * net->layers[hidden_layer_idx].weights[weight_index];
    }

    // Apply ReLU derivative (1 if value > 0, 0 otherwise)
    if (net->layers[hidden_layer_idx].values[i] <= 0) {
      error_sum = 0.0;
    }

    hidden_errors[hidden_layer_idx - 1][i] = error_sum;
  }

  // Backpropagate through remaining hidden layers (if any)
  for (int layer = hidden_layer_idx - 1; layer > 0; layer--) {
    for (int i = 0; i < net->layers[layer].size; i++) {
      float error_sum = 0.0;

      // Sum contributions from next layer
      for (int j = 0; j < net->layers[layer + 1].size; j++) {
        int weight_index = i * net->layers[layer + 1].size + j;
        error_sum +=
            hidden_errors[layer][j] * net->layers[layer].weights[weight_index];
      }

      // Apply ReLU derivative
      if (net->layers[layer].values[i] <= 0) {
        error_sum = 0.0;
      }

      hidden_errors[layer - 1][i] = error_sum;
    }
  }

  // Update weights and biases for each layer
  // Start with output layer
  for (int j = 0; j < net->layers[output_layer_idx].size; j++) {
    // Update weights between last hidden and output layer
    for (int i = 0; i < net->layers[hidden_layer_idx].size; i++) {
      int weight_index = i * net->layers[output_layer_idx].size + j;

      // Gradient = output_error * hidden_activation
      float gradient =
          output_error[j] * net->layers[hidden_layer_idx].values[i];

      // Update weight with gradient descent
      net->layers[hidden_layer_idx].weights[weight_index] -=
          learning_rate * gradient;
    }

    // Update bias for output neuron
    net->layers[hidden_layer_idx].biases[j] -= learning_rate * output_error[j];
  }

  // Update weights and biases for hidden layers
  for (int layer = hidden_layer_idx - 1; layer >= 0; layer--) {
    for (int j = 0; j < net->layers[layer + 1].size; j++) {
      // Update weights
      for (int i = 0; i < net->layers[layer].size; i++) {
        int weight_index = i * net->layers[layer + 1].size + j;

        // Calculate gradient
        float gradient = hidden_errors[layer][j] * net->layers[layer].values[i];

        // Update weight
        net->layers[layer].weights[weight_index] -= learning_rate * gradient;
      }

      // Update bias
      net->layers[layer].biases[j] -= learning_rate * hidden_errors[layer][j];
    }
  }

  // Free allocated memory
  free(output_error);
  for (int layer = 0; layer < output_layer_idx - 1; layer++) {
    free(hidden_errors[layer]);
  }
  free(hidden_errors);
}

/**
 * Train the network on a dataset using mini-batch gradient descent.
 * Performs multiple passes (epochs) through the training data,
 * updating weights after each mini-batch to minimize loss.
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
                   int batch_size) {
  printf("Starting training for %d epochs with learning rate %.4f\n",
         num_epochs, learning_rate);

  // Allocate memory for tracking which samples to use in each batch
  int *indices = malloc(num_train_samples * sizeof(int));
  if (!indices) {
    printf("Error: Failed to allocate memory for training indices\n");
    return;
  }

  // Initialize indices
  for (int i = 0; i < num_train_samples; i++) {
    indices[i] = i;
  }

  // Training loop - iterate through epochs
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    float total_loss = 0.0;
    int correct_predictions = 0;

    // Shuffle indices for randomized training
    shuffle_indices(indices, num_train_samples);

    // Process in mini-batches
    for (int batch_start = 0; batch_start < num_train_samples;
         batch_start += batch_size) {
      int batch_end = batch_start + batch_size;
      if (batch_end > num_train_samples) {
        batch_end = num_train_samples;
      }

      // Process each sample in the batch
      for (int i = batch_start; i < batch_end; i++) {
        int sample_idx = indices[i];

        // Perform backward pass (includes forward pass)
        backward_pass(net, train_images[sample_idx], train_labels[sample_idx],
                      learning_rate);

        // Calculate loss for monitoring
        forward_pass(net, train_images[sample_idx]);
        float loss = cross_entropy_loss(net->layers[net->num_layers - 1].values,
                                        train_labels[sample_idx]);
        total_loss += loss;

        // Determine if prediction was correct
        int predicted_digit = 0;
        float max_prob = net->layers[net->num_layers - 1].values[0];

        for (int j = 1; j < OUTPUTS_SIZE; j++) {
          if (net->layers[net->num_layers - 1].values[j] > max_prob) {
            max_prob = net->layers[net->num_layers - 1].values[j];
            predicted_digit = j;
          }
        }

        if (predicted_digit == train_labels[sample_idx]) {
          correct_predictions++;
        }
      }

      // Print progress after every 1000 samples
      if ((batch_end % 1000) == 0 || batch_end == num_train_samples) {
        printf("Epoch %d: Processed %d/%d samples\r", epoch + 1, batch_end,
               num_train_samples);
        fflush(stdout);
      }
    }

    // Calculate and print epoch metrics
    float avg_loss = total_loss / num_train_samples;
    float accuracy = 100.0f * correct_predictions / num_train_samples;

    printf("\nEpoch %d/%d - Loss: %.4f - Accuracy: %.2f%%\n", epoch + 1,
           num_epochs, avg_loss, accuracy);
  }

  free(indices);
  printf("Training completed.\n");
}

/**
 * Shuffle array of indices using Fisher-Yates algorithm.
 * Used to randomize the order of training examples in each epoch.
 *
 * @param indices Array of indices to shuffle
 * @param count   Number of indices
 */
void shuffle_indices(int *indices, int count) {
  for (int i = 0; i < count - 1; i++) {
    // Generate random index j such that i <= j < count
    int j = i + rand() % (count - i);

    // Swap indices[i] and indices[j]
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
}
