/**
 * mnist_loader.c
 *
 * Implementation of MNIST dataset loading functions.
 * Handles loading and parsing of the MNIST binary file format for
 * both images and labels, with memory management functions.
 */
#include "mnist_loader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Load MNIST images from binary file into memory.
 *
 * @param filename   Path to the MNIST images file
 * @param images     Pointer to array where image data will be stored
 * @param num_images Pointer to store the number of images loaded
 * @return           0 on success, 1 on failure
 */
int load_mnist_images(const char *filename, float ***images, int *num_images) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    printf("Error: Could not open file %s\n", filename);
    return 1;
  }

  // Read header fields (per MNIST file format specification)
  uint32_t magic_number;
  uint32_t n_images;
  uint32_t n_rows;
  uint32_t n_cols;

  // Read magic number (expected value: 2051 for images file)
  fread(&magic_number, sizeof(uint32_t), 1, file);
  magic_number =
      __builtin_bswap32(magic_number); // Convert from big-endian to host endian
  if (magic_number != 2051) {
    printf("Error: Invalid magic number in %s (expected 2051, got %u)\n",
           filename, magic_number);
    fclose(file);
    return 1;
  }

  // Read number of images, rows, and columns
  fread(&n_images, sizeof(uint32_t), 1, file);
  fread(&n_rows, sizeof(uint32_t), 1, file);
  fread(&n_cols, sizeof(uint32_t), 1, file);

  // Convert from big-endian to host endian
  n_images = __builtin_bswap32(n_images);
  n_rows = __builtin_bswap32(n_rows);
  n_cols = __builtin_bswap32(n_cols);

  // Calculate total pixels per image
  int img_size = (int)(n_rows * n_cols); // Cast to int for later use

  // Store the number of images as int for return value
  int num_img = (int)n_images; // Cast to int for consistency

  // Allocate memory for image data using two-level pointer structure
  // First allocate array of pointers to images
  *images = (float **)malloc(num_img * sizeof(float *));
  if (!*images) {
    printf("Error: Failed to allocate memory for image pointers\n");
    fclose(file);
    return 1;
  }

  // Allocate memory for each individual image
  for (int i = 0; i < num_img; i++) {
    (*images)[i] = (float *)malloc(img_size * sizeof(float));
    if (!(*images)[i]) {
      printf("Error: Failed to allocate memory for image %d\n", i);
      // Free previously allocated memory to prevent leaks
      for (int j = 0; j < i; j++) {
        free((*images)[j]);
      }
      free(*images);
      fclose(file);
      return 1;
    }
  }

  // Read raw pixel data and normalize to [0,1] range
  unsigned char pixel;
  for (int i = 0; i < num_img; i++) {
    for (int j = 0; j < img_size; j++) {
      fread(&pixel, sizeof(unsigned char), 1, file);
      // Normalize pixel value from [0,255] to [0,1] range for neural network
      (*images)[i][j] = pixel / 255.0f;
    }
  }

  // Set the number of images output parameter
  *num_images = num_img;

  fclose(file);
  return 0;
}

/**
 * Load MNIST labels from binary file into memory.
 *
 * @param filename    Path to the MNIST labels file
 * @param labels      Pointer to array where label data will be stored
 * @param num_labels  Pointer to store the number of labels loaded
 * @return            0 on success, 1 on failure
 */
int load_mnist_labels(const char *filename, int **labels, int *num_labels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    printf("Error: Could not open file %s\n", filename);
    return 1;
  }

  // Read header fields (per MNIST file format specification)
  uint32_t magic_number;
  uint32_t n_labels;

  // Read magic number (expected value: 2049 for labels file)
  fread(&magic_number, sizeof(uint32_t), 1, file);
  magic_number =
      __builtin_bswap32(magic_number); // Convert from big-endian to host endian
  if (magic_number != 2049) {
    printf("Error: Invalid magic number in %s (expected 2049, got %u)\n",
           filename, magic_number);
    fclose(file);
    return 1;
  }

  // Read number of labels
  fread(&n_labels, sizeof(uint32_t), 1, file);
  n_labels =
      __builtin_bswap32(n_labels); // Convert from big-endian to host endian

  // Convert to int for consistency with other functions
  int num_lbl = (int)n_labels;

  // Allocate memory for labels
  *labels = (int *)malloc(num_lbl * sizeof(int));
  if (!*labels) {
    printf("Error: Failed to allocate memory for labels\n");
    fclose(file);
    return 1;
  }

  // Read label data (single byte per label)
  unsigned char label;
  for (int i = 0; i < num_lbl; i++) {
    fread(&label, sizeof(unsigned char), 1, file);
    (*labels)[i] = (int)label;
  }

  // Set the number of labels output parameter
  *num_labels = num_lbl;

  fclose(file);
  return 0;
}

/**
 * Free memory allocated for images.
 *
 * @param images     Pointer to array of image data
 * @param num_images Number of images to free
 */
void free_mnist_images(float **images, int num_images) {
  if (images) {
    for (int i = 0; i < num_images; i++) {
      free(images[i]);
    }
    free(images);
  }
}

/**
 * Free memory allocated for labels.
 *
 * @param labels Pointer to array of label data
 */
void free_mnist_labels(int *labels) { free(labels); }
