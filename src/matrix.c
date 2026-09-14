#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

Matrix matrix_create(int rows,int cols){
    Matrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = malloc(rows * cols * sizeof(float));

    return matrix;
}

void matrix_set(Matrix *matrix,int row,int col, float value){
    int index = row * matrix->cols + col;
    matrix->data[index] = value;
}

void matrix_print(Matrix *matrix){
    for(int i=0;i < matrix->rows;i++){
        for (int j = 0;j<matrix->cols;j++)
        {
            int index = i*matrix->cols + j;
            printf("%.2f ", matrix->data[index]);
        }
        printf("\n");
    }
}

float matrix_get(Matrix *matrix,int row,int col){
    int index = row * matrix->cols + col;
    return matrix->data[index];
}

void matrix_free(Matrix *matrix){
    free(matrix->data);
    matrix->data = NULL;
}

Matrix matrix_add(Matrix *a, Matrix *b){
    Matrix c = matrix_create(a->rows,a->cols);
    for(int i = 0;i < a->rows;i++){
        for(int j = 0;j<a->cols;j++){
            matrix_set(&c,
                i,
                j,
                matrix_get(a,i,j)+matrix_get(b,i,j)
            );
        }
    }
    return c;
}

Matrix matrix_mul(Matrix *a, Matrix *b){
    if(a->cols != b->rows){
        printf("Mtricele nu pot fi inmultite!\n");
        Matrix empty = {0,0,NULL};
        return empty;

    }
    Matrix c = matrix_create(a->rows,b->cols);
    for (int i = 0; i<a->rows;i++){
        for(int j = 0; j < b ->cols;j++){
            float sum = 0;
            for(int k = 0; k<a->cols;k++){
                sum += matrix_get(a,i,k)*matrix_get(b,k,j);
            }
            matrix_set(&c,i,j,sum);
        }
    }
    return c;
}