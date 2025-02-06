#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int num_inputs = 2; // Number of input neurons
int num_hidden = 4; // Number of hidden neurons

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

// Activation functions and their derivatives
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double x) { return 1 - tanh(x) * tanh(x); }

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double z) { return z > 0 ? 1.0 : 0.0; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double forward_pass(double *x, double **w1, double *b1, double *w2, double b2,
                    double *z, double *h) {
  for (int i = 0; i < num_hidden; i++) {
    z[i] = b1[i]; // Start with bias
    for (int j = 0; j < num_inputs; j++) {
      z[i] += w1[i][j] * x[j]; // Weighted sum
    }
    h[i] = tanh_activation(z[i]); // Apply activation function
  }

  // Compute final output
  double z_output = b2;
  for (int i = 0; i < num_hidden; i++) {
    z_output += w2[i] * h[i];
  }

  // Apply sigmoid activation for binary output
  return sigmoid(z_output);
}

// Training data for XOR
double training_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double targets[4] = {0, 1, 1, 0};

// Binary Cross-Entropy Loss function
double compute_loss(double y_true, double y_pred) {
  // Clamp y_pred to avoid log(0)
  if (y_pred <= 1e-15) {
    y_pred = 1e-15;
  }
  if (y_pred >= 1 - 1e-15) {
    y_pred = 1 - 1e-15;
  }

  return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred));
}

// Backpropagation to update weights
void backpropagate(double *x, double y_true, double **w1, double *b1,
                   double *w2, double *b2, double learning_rate, double *loss) {
  double *z = malloc(num_hidden * sizeof(double));
  double *h = malloc(num_hidden * sizeof(double));
  double *dL_dw2 = malloc(num_hidden * sizeof(double));
  double *dL_db1 = malloc(num_hidden * sizeof(double));
  double **dL_dw1 = malloc(num_hidden * sizeof(double *));

  for (int i = 0; i < num_hidden; i++) {
    dL_dw1[i] = malloc(num_inputs * sizeof(double));
  }

  double y_pred = forward_pass(x, w1, b1, w2, *b2, z, h);
  *loss = compute_loss(y_true, y_pred);

  double dL_dz_output = y_pred - y_true;

  // Compute gradients for w2 and b2
  for (int i = 0; i < num_hidden; i++) {
    dL_dw2[i] = dL_dz_output * h[i];
    w2[i] -= learning_rate * dL_dw2[i];
  }
  *b2 -= learning_rate * dL_dz_output;

  // Compute gradients for w1 and b1
  for (int i = 0; i < num_hidden; i++) {
    double dh = dL_dz_output * w2[i];
    double dz = dh * tanh_derivative(z[i]);

    for (int j = 0; j < num_inputs; j++) {
      dL_dw1[i][j] = dz * x[j];
      w1[i][j] -= learning_rate * dL_dw1[i][j];
    }

    dL_db1[i] = dz;
    b1[i] -= learning_rate * dL_db1[i];
  }

  // Free allocated memory
  free(z);
  free(h);
  free(dL_dw2);
  free(dL_db1);
  for (int i = 0; i < num_hidden; i++) {
    free(dL_dw1[i]);
  }
  free(dL_dw1);
}

// Evaluate the trained network on the XOR dataset
void evaluate_network(double **w1, double *b1, double *w2, double b2) {
  double *z = malloc(num_hidden * sizeof(double));
  double *h = malloc(num_hidden * sizeof(double));

  printf("\nNetwork Evaluation:\n");
  for (int i = 0; i < 4; i++) {
    double pred = forward_pass(training_data[i], w1, b1, w2, b2, z, h);
    printf("Input: [%.0f, %.0f], Target: %.0f, Prediction: %f\n",
           training_data[i][0], training_data[i][1], targets[i], pred);
  }

  free(z);
  free(h);
}

int main() {
  srand(time(NULL));

  // Allocate memory for weights and biases
  double **w1 = malloc(num_hidden * sizeof(double *));
  double *b1 = malloc(num_hidden * sizeof(double));
  double *w2 = malloc(num_hidden * sizeof(double));
  double b2 = 0;

  for (int i = 0; i < num_hidden; i++) {
    w1[i] = malloc(num_inputs * sizeof(double));
  }

  // Initialize weights and biases
  for (int i = 0; i < num_hidden; i++) {
    for (int j = 0; j < num_inputs; j++) {
      w1[i][j] = he_init(num_inputs);
    }
    w2[i] = he_init(num_hidden);
    b1[i] = 0;
  }

  // Print initial weights and biases
  printf("Initial Weights and Biases:\n");
  for (int i = 0; i < num_hidden; i++) {
    for (int j = 0; j < num_inputs; j++) {
      printf("w1[%d][%d]: %f\n", i, j, w1[i][j]);
    }
  }
  for (int i = 0; i < num_hidden; i++) {
    printf("b1[%d]: %f\n", i, b1[i]);
  }
  for (int i = 0; i < num_hidden; i++) {
    printf("w2[%d]: %f\n", i, w2[i]);
  }
  printf("b2: %f\n\n", b2);

  int epochs = 10000;
  double learning_rate = 0.05;

  // Training loop with early stopping
  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0;

    for (int i = 0; i < 4; i++) {
      double loss;
      backpropagate(training_data[i], targets[i], w1, b1, w2, &b2,
                    learning_rate, &loss);
      total_loss += loss;
    }

    if ((epoch + 1) % 100 == 0) {
      printf("Epoch %d, Average Loss: %f\n", epoch + 1, total_loss / 4);
      printf("Sample weights - w1[0][0]: %f, w2[0]: %f\n", w1[0][0], w2[0]);
    }

    if (total_loss / 4 < 0.01) {
      printf("Stopping early at epoch %d, loss: %f\n", epoch + 1,
             total_loss / 4);
      break;
    }
  }

  evaluate_network(w1, b1, w2, b2);

  // Free allocated memory
  for (int i = 0; i < num_hidden; i++) {
    free(w1[i]);
  }
  free(w1);
  free(b1);
  free(w2);

  return 0;
}
