#include <stdio.h>
#include "neural_network.h"
#include "activation.h"
#include <stdlib.h>
#include <time.h>

int main() {
    srand(5);

    Neuron neuron1 = neuron_create();
    Neuron neuron2 = neuron_create();
    Neuron neuron3 = neuron_create();

    float x1[] = {0,0,1,1};
    float x2[] = {0,1,0,1};
    float target[] = {0,1,1,0};

    for(int i=0;i<10000;i++) {
        float total_loss = 0.0f;

        for(int j=0;j<4;j++) {
            float output_n1 = neuron_forward(&neuron1,x1[j],x2[j]);
            float output_n2 = neuron_forward(&neuron2,x1[j],x2[j]);
            float output_n3 = neuron_forward(&neuron3,output_n1,output_n2);

            float error = output_n3 - target[j];
            float loss = 0.5f * error * error;
            total_loss += loss;

            float z3 = neuron3.w1*output_n1 + neuron3.w2*output_n2 + neuron3.bias;
            float relu_gradient = (z3 > 0.0f) ? 1.0f : 0.01f;

            float gradient_z3 = error * relu_gradient;
            float gradient_h1 = gradient_z3 * neuron3.w1;
            float gradient_h2 = gradient_z3 * neuron3.w2;

            neuron_train(&neuron1,x1[j],x2[j],gradient_h1,0.01f);
            neuron_train(&neuron2,x1[j],x2[j],gradient_h2,0.01f);
            neuron_train(&neuron3,output_n1,output_n2,error,0.01f);
        }

        if(i%1000==0) {
            printf("Epoca:%d | loss:%.6f\n",i,total_loss/4);
            printf("N1:w1=%.4f w2=%.4f b=%.4f\n",neuron1.w1,neuron1.w2,neuron1.bias);
            printf("N2:w1=%.4f w2=%.4f b=%.4f\n",neuron2.w1,neuron2.w2,neuron2.bias);
            printf("N3:w1=%.4f w2=%.4f b=%.4f\n\n",neuron3.w1,neuron3.w2,neuron3.bias);
        }
    }

    printf("\n========== TEST ==========\n");
    for(int j=0;j<4;j++) {
        float output_n1 = neuron_forward(&neuron1,x1[j],x2[j]);
        float output_n2 = neuron_forward(&neuron2,x1[j],x2[j]);
        float output_n3 = neuron_forward(&neuron3,output_n1,output_n2);
        printf("%.0f XOR %.0f = %.4f\n",x1[j],x2[j],output_n3);
    }
    return 0;
}
