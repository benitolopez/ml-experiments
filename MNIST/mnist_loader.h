/**
 * mnist_loader.h
 *
 * Header for MNIST dataset loading functionality.
 * Provides functions to load, parse, and manage MNIST dataset files.
 */
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

/**
 * Load MNIST images from binary file into memory.
 * Parses the IDX file format and converts images to float arrays.
 *
 * @param filename   Path to the MNIST images file
 * @param images     Pointer to array where image data will be stored
 * @param num_images Pointer to store the number of images loaded
 * @return           0 on success, 1 on failure
 */
int load_mnist_images(const char *filename, float ***images, int *num_images);

/**
 * Load MNIST labels from binary file into memory.
 * Parses the IDX file format and loads the digit labels.
 *
 * @param filename    Path to the MNIST labels file
 * @param labels      Pointer to array where label data will be stored
 * @param num_labels  Pointer to store the number of labels loaded
 * @return            0 on success, 1 on failure
 */
int load_mnist_labels(const char *filename, int **labels, int *num_labels);

/**
 * Free memory allocated for images.
 *
 * @param images     Pointer to array of image data
 * @param num_images Number of images to free
 */
void free_mnist_images(float **images, int num_images);

/**
 * Free memory allocated for labels.
 *
 * @param labels Pointer to array of label data
 */
void free_mnist_labels(int *labels);

#endif // MNIST_LOADER_H
