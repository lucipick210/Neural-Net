#include <math.h>
#include "activation.h"

float relu(float x){
    if(x>0){
        return x;
    }else{
        return 0;
    }
}

float sigmoid(float x){
    return 1/(1+exp(-x));
}

float tanh_activation(float x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

Matrix matrix_relu(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);
            value = relu(value);
            
            matrix_set(&result,i,j,value);

        }
    }
    return result;
}


Matrix matrix_sigmoid(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);
            value = sigmoid(value);
            
            matrix_set(&result,i,j,value);

        }
    }
    return result;
}

Matrix matrix_tanh(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);
            value = tanh_activation(value);
            
            matrix_set(&result,i,j,value);

        }
    }
    return result;
}