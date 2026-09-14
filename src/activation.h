#include "matrix.h"
#ifndef ACTIVATION_H
#define ACTIVATION_H
float relu(float x);
float sigmoid(float x);
float tanh_activation(float x);
Matrix matrix_relu(Matrix *matrix);
Matrix matrix_sigmoid(Matrix *matrix);
Matrix matrix_tanh(Matrix *matrix);
#endif