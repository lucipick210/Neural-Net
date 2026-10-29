#ifndef MATRIX_H
#define MATRIX_H

typedef struct{
    int rows;
    int cols;
    float *data;
}Matrix;

Matrix matrix_create(int rows, int cols);
void matrix_set(Matrix *matrix, int row, int col, float value);
void matrix_print(Matrix *matrix);
void matrix_free(Matrix *matrix);
float matrix_get(Matrix *matrix,int row,int col);
Matrix matrix_add(Matrix *a,Matrix *b);
Matrix matrix_mul(Matrix *a,Matrix *b);
#endif

