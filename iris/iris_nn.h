#ifndef IRIS_NN_H
#define IRIS_NN_H

#define NUM_INPUTS 4
#define NUM_HIDDEN 8
#define NUM_OUTPUTS 3

void train_iris_nn(double dataset[][4], int labels[], int num_samples,
                   double ***w1, double **b1, double ***w2, double **b2);

void evaluate_iris_nn(double dataset[][4], int labels[], int num_samples,
                      double **w1, double *b1, double **w2, double *b2);

void load_model(double ***w1, double **b1, double ***w2, double **b2);
void save_model(double **w1, double *b1, double **w2, double *b2);

#endif
