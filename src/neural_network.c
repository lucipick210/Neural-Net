#include "neural_network.h"
#include "activation.h"
#include<stdlib.h>
float neuron_forward(Neuron *neuron,float x1,float x2){
    float z = (neuron->w1*x1)+(neuron->w2*x2)+neuron->bias;
    return leaky_relu(z);
}

Neuron neuron_create(){
    Neuron neuron;

    neuron.w1 =((float)rand()/RAND_MAX)*2.0f-1.0f;
    neuron.w2 =((float)rand()/RAND_MAX)*2.0f-1.0f;
    neuron.bias =((float)rand()/RAND_MAX)*2.0f-1.0f;

    return neuron;
}

void neuron_train(Neuron *neuron,float x1,float x2,float neuron_gradient,float learning_rate){
    float z = (neuron->w1*x1)+(neuron->w2*x2)+neuron->bias;
    float output = neuron_forward(neuron,x1,x2); 
    // float error = target - output;
    float relu_gradient;
    if (z>0.0f){
        relu_gradient = 1.0f;
    }else{
        relu_gradient = 0.01f;
    }
    float gradient_z = neuron_gradient*relu_gradient;
    float gradient_w1 = gradient_z * x1;
    float gradient_w2 = gradient_z * x2;
    float gradient_bias = gradient_z;
    neuron->w1-=learning_rate*gradient_w1;
    neuron->w2-=learning_rate*gradient_w2;
    neuron->bias-=learning_rate*gradient_bias;
}





