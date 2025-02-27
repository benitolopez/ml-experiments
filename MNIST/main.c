// main.c
#include "mnist_loader.h"
#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Weights file path
#define WEIGHTS_FILE "build/network_weights.bin"

int main() {
  // Seed the random number generator
  srand(time(NULL));

  // Create network with 784 inputs (28x28 image), 128 hidden neurons, and 10
  // outputs (digits 0-9)
  int layer_sizes[] = {INPUTS_SIZE, 128, OUTPUTS_SIZE};
  struct Network *net = create_network(layer_sizes, 3);

  if (net == NULL) {
    printf("Failed to create network\n");
    return 1;
  }

  // Try to load weights from file
  int trained = 0;
  FILE *weights_file = fopen(WEIGHTS_FILE, "rb");
  if (weights_file != NULL) {
    fclose(weights_file);

    printf("Found existing weights file. Loading trained model...\n");
    if (load_network(net, WEIGHTS_FILE) == 0) {
      trained = 1;
    } else {
      printf("Error loading weights. Will retrain the network.\n");
    }
  }

  // If weights couldn't be loaded, train the network
  if (!trained) {
    // Load MNIST training data
    float **train_images;
    int *train_labels;
    int num_train_images, num_train_labels;

    printf("Loading MNIST training data...\n");

    if (load_mnist_images("data/train-images.idx3-ubyte", &train_images,
                          &num_train_images) != 0) {
      free_network(net);
      return 1;
    }

    if (load_mnist_labels("data/train-labels.idx1-ubyte", &train_labels,
                          &num_train_labels) != 0) {
      free_mnist_images(train_images, num_train_images);
      free_network(net);
      return 1;
    }

    printf("Loaded %d training images and %d training labels\n",
           num_train_images, num_train_labels);

    // Train the network
    float learning_rate = 0.01;
    int num_epochs = 5;
    int batch_size = 32;

    train_network(net, train_images, train_labels, num_train_images,
                  learning_rate, num_epochs, batch_size);

    // Save the trained network
    printf("Saving trained network weights...\n");
    save_network(net, WEIGHTS_FILE);

    // Clean up training data
    free_mnist_images(train_images, num_train_images);
    free_mnist_labels(train_labels);
  }

  // Load test data for evaluation
  float **test_images;
  int *test_labels;
  int num_test_images, num_test_labels;

  printf("\nLoading MNIST test data...\n");

  if (load_mnist_images("data/t10k-images.idx3-ubyte", &test_images,
                        &num_test_images) != 0) {
    free_network(net);
    return 1;
  }

  if (load_mnist_labels("data/t10k-labels.idx1-ubyte", &test_labels,
                        &num_test_labels) != 0) {
    free_mnist_images(test_images, num_test_images);
    free_network(net);
    return 1;
  }

  printf("Loaded %d test images and %d test labels\n", num_test_images,
         num_test_labels);

  // Evaluate the network on test data
  evaluate_network(net, test_images, test_labels, num_test_images);

  // Test a few examples individually
  printf("\nTesting random examples:\n");
  for (int i = 0; i < 5; i++) {
    // Pick a random example from the test set
    int random_idx = rand() % num_test_images;

    forward_pass(net, test_images[random_idx]);

    // Find highest probability
    float max_prob = net->layers[2].values[0];
    int predicted_digit = 0;

    for (int j = 1; j < OUTPUTS_SIZE; j++) {
      if (net->layers[2].values[j] > max_prob) {
        max_prob = net->layers[2].values[j];
        predicted_digit = j;
      }
    }

    printf("Example %d - Predicted: %d, Actual: %d, Confidence: %.2f%%\n",
           i + 1, predicted_digit, test_labels[random_idx], max_prob * 100.0);
  }

  // Clean up
  free_mnist_images(test_images, num_test_images);
  free_mnist_labels(test_labels);
  free_network(net);

  return 0;
}
