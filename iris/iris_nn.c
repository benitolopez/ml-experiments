#include "iris_nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// He Initialization for weight initialization
double he_init(int n_in) {
  // Avoid log(0)
  double u1 = ((double)rand() + 1) / (RAND_MAX + 1.0);
  double u2 = ((double)rand() + 1) / (RAND_MAX + 1.0);

  // Box-Muller transform for normal distribution
  double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

  // Scale by He standard deviation
  return z * sqrt(2.0 / n_in);
}

// Activation function
double tanh_activation(double x) { return tanh(x); }

// Softmax activation function for multi-class classification
void softmax(double *z, double *output) {
  double max_val = z[0];

  // Find max value
  for (int i = 1; i < NUM_OUTPUTS; i++) {
    if (z[i] > max_val) {
      max_val = z[i];
    }
  }

  // Compute exponentials with stability
  double sum_exp = 0.0;
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    output[i] = exp(z[i] - max_val); // Subtract max for stability
    sum_exp += output[i];
  }

  // Normalize to get probabilities
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    output[i] /= sum_exp;
  }
}

void forward_pass(double *x, double **w1, double *b1, double **w2, double *b2,
                  double *z_hidden, double *h_hidden, double *z_output,
                  double *y_pred) {
  // Compute hidden layer activations
  for (int i = 0; i < NUM_HIDDEN; i++) {
    z_hidden[i] = b1[i]; // Start with bias
    for (int j = 0; j < NUM_INPUTS; j++) {
      z_hidden[i] += w1[i][j] * x[j]; // Weighted sum
    }
    h_hidden[i] = tanh_activation(z_hidden[i]); // Apply activation function
  }

  // Compute output layer activations
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    z_output[i] = b2[i]; // Bias for each output neuron
    for (int j = 0; j < NUM_HIDDEN; j++) {
      z_output[i] += w2[i][j] * h_hidden[j];
    }
  }

  // Apply softmax
  softmax(z_output, y_pred);
}

// Categorical cross-entropy loss function
double compute_loss(int true_label, double *y_pred) {
  return -log(y_pred[true_label]);
}

// Backpropagation for multi-class classification
void backpropagate(double *x, int y_true, double **w1, double *b1, double **w2,
                   double *b2, double learning_rate, double *loss) {
  double *z_hidden = malloc(NUM_HIDDEN * sizeof(double));
  double *h_hidden = malloc(NUM_HIDDEN * sizeof(double));
  double *z_output = malloc(NUM_OUTPUTS * sizeof(double));
  double *y_pred = malloc(NUM_OUTPUTS * sizeof(double));

  forward_pass(x, w1, b1, w2, b2, z_hidden, h_hidden, z_output, y_pred);
  *loss = compute_loss(y_true, y_pred);

  // Compute gradient for output layer
  double *dL_dz_output = malloc(NUM_OUTPUTS * sizeof(double));
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    dL_dz_output[i] = y_pred[i] - (i == y_true ? 1 : 0); // Softmax gradient
  }

  // Compute gradient for w2 and b2
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      w2[i][j] -= learning_rate * dL_dz_output[i] * h_hidden[j];
    }
    b2[i] -= learning_rate * dL_dz_output[i];
  }

  // Compute gradient for hidden layer
  double *dL_dh = malloc(NUM_HIDDEN * sizeof(double));
  for (int i = 0; i < NUM_HIDDEN; i++) {
    dL_dh[i] = 0;
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      dL_dh[i] += dL_dz_output[j] * w2[j][i];
    }
    dL_dh[i] *= (1 - h_hidden[i] * h_hidden[i]); // Tanh derivative
  }

  // Compute gradient for w1 and b1
  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_INPUTS; j++) {
      w1[i][j] -= learning_rate * dL_dh[i] * x[j];
    }
    b1[i] -= learning_rate * dL_dh[i];
  }

  // Free allocated memory
  free(z_hidden);
  free(h_hidden);
  free(z_output);
  free(y_pred);
  free(dL_dz_output);
  free(dL_dh);
}

void train_iris_nn(double dataset[][4], int labels[], int num_samples,
                   double ***w1, double **b1, double ***w2, double **b2) {
  printf("Training Neural Network on Iris dataset...\n");

  // Allocate memory for weights and biases
  *w1 = malloc(NUM_HIDDEN * sizeof(double *));
  *b1 = malloc(NUM_HIDDEN * sizeof(double));
  *w2 = malloc(NUM_OUTPUTS * sizeof(double *));
  *b2 = malloc(NUM_OUTPUTS * sizeof(double));

  for (int i = 0; i < NUM_HIDDEN; i++) {
    (*w1)[i] = malloc(NUM_INPUTS * sizeof(double));
    for (int j = 0; j < NUM_INPUTS; j++) {
      (*w1)[i][j] = he_init(NUM_INPUTS);
    }
    (*b1)[i] = 0;
  }

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    (*w2)[i] = malloc(NUM_HIDDEN * sizeof(double));
    for (int j = 0; j < NUM_HIDDEN; j++) {
      (*w2)[i][j] = he_init(NUM_HIDDEN);
    }
    (*b2)[i] = 0;
  }

  int epochs = 5000;
  double learning_rate = 0.01;

  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0;
    for (int i = 0; i < num_samples; i++) {
      double loss;
      backpropagate(dataset[i], labels[i], *w1, *b1, *w2, *b2, learning_rate,
                    &loss);
      total_loss += loss;
    }
    if ((epoch + 1) % 100 == 0) {
      printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / num_samples);
    }
  }

  int correct = 0; // Track correct predictions

  for (int i = 0; i < num_samples; i++) {
    double z_hidden[NUM_HIDDEN], h_hidden[NUM_HIDDEN], z_output[NUM_OUTPUTS],
        y_pred[NUM_OUTPUTS];

    forward_pass(dataset[i], *w1, *b1, *w2, *b2, z_hidden, h_hidden, z_output,
                 y_pred);

    // Find the predicted class
    int predicted_class = 0;
    for (int j = 1; j < NUM_OUTPUTS; j++) {
      if (y_pred[j] > y_pred[predicted_class]) {
        predicted_class = j;
      }
    }

    // Compare prediction with the actual label
    if (predicted_class == labels[i]) {
      correct++;
    }
  }

  // Print Training Accuracy
  printf("Training Accuracy: %.2f%%\n", (double)correct / num_samples * 100.0);
}

void evaluate_iris_nn(double dataset[][4], int labels[], int num_samples,
                      double **w1, double *b1, double **w2, double *b2) {
  int correct = 0;

  for (int i = 0; i < num_samples; i++) {
    double z_hidden[NUM_HIDDEN], h_hidden[NUM_HIDDEN], z_output[NUM_OUTPUTS],
        y_pred[NUM_OUTPUTS];

    forward_pass(dataset[i], w1, b1, w2, b2, z_hidden, h_hidden, z_output,
                 y_pred);

    // Find the index of the max probability (predicted class)
    int predicted_class = 0;
    for (int j = 1; j < NUM_OUTPUTS; j++) {
      if (y_pred[j] > y_pred[predicted_class]) {
        predicted_class = j;
      }
    }

    if (predicted_class == labels[i]) {
      correct++;
    }
  }

  double accuracy = (double)correct / num_samples * 100.0;
  printf("Model Accuracy: %.2f%%\n", accuracy);
}

void save_model(double **w1, double *b1, double **w2, double *b2) {
  FILE *file = fopen("model.txt", "w");

  if (file == NULL) {
    printf("Error: Cannot open file.\n");
    return;
  }

  fprintf(file, "# w1\n");
  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_INPUTS; j++) {
      fprintf(file, "%lf\n", w1[i][j]);
    }
  }

  fprintf(file, "# b1\n");
  for (int i = 0; i < NUM_HIDDEN; i++) {
    fprintf(file, "%lf\n", b1[i]);
  }

  fprintf(file, "# w2\n");
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      fprintf(file, "%lf\n", w2[i][j]);
    }
  }

  fprintf(file, "# b2\n");
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    fprintf(file, "%lf\n", b2[i]);
  }

  fclose(file);
}

void skip_comments(FILE *file) {
  char buffer[50];
  while (fgets(buffer, sizeof(buffer), file)) {
    if (buffer[0] != '#' && buffer[0] != '\n') {
      fseek(file, -strlen(buffer), SEEK_CUR); // Rewind to valid data
      break;
    }
  }
}

void load_model(double ***w1, double **b1, double ***w2, double **b2) {
  FILE *file = fopen("model.txt", "r");

  if (file == NULL) {
    printf("Error: Cannot open model file.\n");
    return;
  }

  // Allocate memory for weights and biases
  *w1 = malloc(NUM_HIDDEN * sizeof(double *));
  *b1 = malloc(NUM_HIDDEN * sizeof(double));
  *w2 = malloc(NUM_OUTPUTS * sizeof(double *));
  *b2 = malloc(NUM_OUTPUTS * sizeof(double));

  for (int i = 0; i < NUM_HIDDEN; i++) {
    (*w1)[i] = malloc(NUM_INPUTS * sizeof(double));
  }

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    (*w2)[i] = malloc(NUM_HIDDEN * sizeof(double));
  }

  // Read w1 (NUM_HIDDEN x NUM_INPUTS)
  skip_comments(file);

  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_INPUTS; j++) {
      fscanf(file, "%lf", &(*w1)[i][j]);
    }
  }

  // Read b1 (NUM_HIDDEN)
  skip_comments(file);

  for (int i = 0; i < NUM_HIDDEN; i++) {
    fscanf(file, "%lf", &(*b1)[i]);
  }

  // Read w2 (NUM_OUTPUTS x NUM_HIDDEN)
  skip_comments(file);

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      fscanf(file, "%lf", &(*w2)[i][j]);
    }
  }

  // Read b2 (NUM_OUTPUTS)
  skip_comments(file);

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    fscanf(file, "%lf", &(*b2)[i]);
  }

  fclose(file);
  printf("Model loaded successfully!\n");
}
