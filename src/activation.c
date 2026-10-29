#include <math.h>
#include "activation.h"

float relu(float x){
    if(x>0){
        return x;
    }else{
        return 0.0f;
    }
}

float sigmoid(float x){
    return 1/(1+exp(-x));
}

float tanh_activation(float x){
    return tanhf(x);
}

Matrix matrix_relu(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);
            
            matrix_set(&result,i,j,relu(value));

        }
    }
    return result;
}


Matrix matrix_sigmoid(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);            
            matrix_set(&result,i,j,sigmoid(value));

        }
    }
    return result;
}

Matrix matrix_tanh(Matrix *matrix){
    Matrix result = matrix_create(matrix->rows,matrix->cols);
    for(int i = 0;i< matrix->rows;i++){
        for(int j = 0;j<matrix->cols;j++){
            float value = matrix_get(matrix,i,j);

            
            matrix_set(&result,i,j,tanh_activation(value));

        }
    }
    return result;
}

float leaky_relu(float x){
    if(x<=0){
        return x*0.01f;
    }else{
        return x;
    }
}


void softmax(float *inputs,float *outputs,int count){
    float max = inputs[0];

    for(int i=1;i<count;i++){
        if(inputs[i]>max)
            max=inputs[i];
    }

    float sum=0.0f;

    for(int i=0;i<count;i++){
        outputs[i]=expf(inputs[i]-max);
        sum+=outputs[i];
    }

    if(!isfinite(sum) || sum<=0.0f){
        for(int i=0;i<count;i++)
            outputs[i]=0.0f;

        return;
    }

    for(int i=0;i<count;i++){
        outputs[i]/=sum;
    }
}


float cross_entropy(float *prediction,float *targets,int count){
    float loss = 0.0f;
    for(int i = 0;i<count;i++){
        float p = prediction[i];
        if(p < 1e-7f){
            p = 1e-7f;
        }

        loss-=targets[i]*logf(p);
    }
    return loss;
}