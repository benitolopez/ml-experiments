#include "iris_nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 150

int main() {
  srand(time(NULL));

  FILE *file = fopen("data/iris.data", "r");
  if (file == NULL) {
    printf("Error: Cannot read the dataset file.\n");
    return 1;
  }

  char line[40];

  double dataset[MAX_SAMPLES][4];
  int labels[MAX_SAMPLES];

  int rows = 0;

  // Read the sample file and store the data
  while (fgets(line, sizeof(line), file) != NULL) {
    double sepal_length, sepal_width, petal_length, petal_width;
    char class_label[20];
    int label;

    sscanf(line, "%lf, %lf, %lf, %lf, %s", &sepal_length, &sepal_width,
           &petal_length, &petal_width, class_label);

    if (strcmp(class_label, "Iris-setosa") == 0) {
      label = 0;
    } else if (strcmp(class_label, "Iris-versicolor") == 0) {
      label = 1;
    } else {
      label = 2;
    }

    dataset[rows][0] = sepal_length;
    dataset[rows][1] = sepal_width;
    dataset[rows][2] = petal_length;
    dataset[rows][3] = petal_width;

    labels[rows] = label;

    if (rows >= MAX_SAMPLES) {
      printf("Warning: Exceeded dataset size!\n");
      break;
    }

    rows++;
  }

  fclose(file);

  // Fisher–Yates shuffle
  for (int i = rows - 1; i > 0; i--) {
    int j = rand() % (i + 1);

    for (int k = 0; k < 4; k++) {
      double temp = dataset[i][k];
      dataset[i][k] = dataset[j][k];
      dataset[j][k] = temp;
    }

    int temp_label = labels[i];
    labels[i] = labels[j];
    labels[j] = temp_label;
  }

  int train_size = rows * 0.8;
  int test_size = rows - train_size;

  double train_dataset[train_size][4];
  int train_labels[train_size];
  double test_dataset[test_size][4];
  int test_labels[test_size];

  for (int i = 0; i < train_size; i++) {
    for (int j = 0; j < 4; j++) {
      train_dataset[i][j] = dataset[i][j];
    }
    train_labels[i] = labels[i];
  }

  for (int i = 0; i < test_size; i++) {
    for (int j = 0; j < 4; j++) {
      test_dataset[i][j] = dataset[train_size + i][j];
    }
    test_labels[i] = labels[train_size + i];
  }

  // Allocate pointers for weights and biases
  double **w1, *b1, **w2, *b2;

  FILE *model_file = fopen("model.txt", "r");

  if (model_file != NULL) {
    fclose(model_file);
    printf("Loading existing model... \n");
    load_model(&w1, &b1, &w2, &b2);
  } else {
    printf("No saved model found. Training a new mode... \n");
    // Train the network
    train_iris_nn(train_dataset, train_labels, train_size, &w1, &b1, &w2, &b2);
    save_model(w1, b1, w2, b2);
  }

  // Evaluate the network
  evaluate_iris_nn(test_dataset, test_labels, test_size, w1, b1, w2, b2);

  // Free dynamically allocated memory
  for (int i = 0; i < NUM_HIDDEN; i++) {
    free(w1[i]);
  }
  free(w1);
  free(b1);

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    free(w2[i]);
  }
  free(w2);
  free(b2);

  return 0;
}
