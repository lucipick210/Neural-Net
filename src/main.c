#include <stdio.h>
// #include "matrix.h"
// #include "activation.h"
#include "neural_network.h"
int main()
{
    // Matrix a = matrix_create(2, 2);
    // Matrix b = matrix_create(2, 2);
    // matrix_set(&a,0,0,1);
    // matrix_set(&a,0,1,2);
    // matrix_set(&a,1,0,3);
    // matrix_set(&a,1,1,4);
    
    // matrix_set(&b,0,0,5);
    // matrix_set(&b,0,1,6);
    // matrix_set(&b,1,0,7);
    // matrix_set(&b,1,1,8);
    // Matrix c = matrix_mul(&a,&b);
    // if(c.data == NULL){
    //     printf("Eroare la inmultire.\n");
    // }
    // Matrix suma = matrix_add(&a,&b);
    // printf("Numarul: %.2f\n",matrix_get(&a,1,0));
    // matrix_print(&c);
    // printf("\nSuma:\n");
    // matrix_print(&suma);

    // matrix_free(&a);
    // matrix_free(&b);
    // matrix_free(&c);
    // matrix_free(&suma);
    // printf("Sigmoid(-5)%.5f\n",sigmoid(-5));
    // printf("Sigmoid(0)%.5f\n",sigmoid(0));
    // printf("Sigmoid(5) %.5f\n",sigmoid(5));
    // printf("Tanh(-5) = %.5f\n", tanh_activation(-5));
    // printf("Tanh(0) = %.5f\n", tanh_activation(0));
    // printf("Tanh(5) = %.5f\n", tanh_activation(5));
    // Matrix a = matrix_create(2, 3);

    // matrix_set(&a, 0, 0, 1.5);
    // matrix_set(&a, 0, 1, -2.0);
    // matrix_set(&a, 0, 2, 3.0);

    // matrix_set(&a, 1, 0, -4.0);
    // matrix_set(&a, 1, 1, 0.5);
    // matrix_set(&a, 1, 2, -1.0);

    // Matrix b = matrix_sigmoid(&a);
    // matrix_print(&a);
    // printf("\n\n");
    // matrix_print(&b);
    // matrix_free(&a);
    // matrix_free(&b);
    Neuron neuron1;

    neuron1.w1 = 0.5;
    neuron1.w2 = 0.2;
    neuron1.bias = 0.1;

    float output = neuron_forward(&neuron1, 2.0, 3.0);

    printf("Neuron output: %.2f\n", output);
    neuron1.w1 = -1.0;
    neuron1.w2 = -1.0;
    neuron1.bias = 0.0;

    output = neuron_forward(&neuron1, 2.0, 3.0);

    printf("Neuron output: %.2f\n", output);        


    Neuron neuron = neuron_create();

    printf("w1 = %.3f\n", neuron.w1);
    printf("w2 = %.3f\n", neuron.w2);
    printf("bias = %.3f\n\n", neuron.bias);
    for(int i = 0;i<=6;i++){
        neuron_train(&neuron,0,0,0,0.5);
        neuron_train(&neuron,0,1,0,0.5);
        neuron_train(&neuron,1,0,0,0.5);
        neuron_train(&neuron,1,1,1,0.5);
        printf("w1 = %.3f\n", neuron.w1);
        printf("w2 = %.3f\n", neuron.w2);
        printf("bias = %.3f\n", neuron.bias);
    }
    float output1 = neuron_forward(&neuron,0,0);
    float output2 = neuron_forward(&neuron,0,1);
    float output3 = neuron_forward(&neuron,1,0);
    float output4 = neuron_forward(&neuron,1,1);
    printf("Neuron1 output: %.2f\n",output1);
    printf("Neuron2 output: %.2f\n",output2);
    printf("Neuron3 output: %.2f\n",output3);
    printf("Neuron4 output: %.2f\n",output4);   
    return 0;
}