#include "neural_network.h"
#include "activation.h"
#include<stdlib.h>
float neuron_forward(Neuron *neuron,float x1,float x2){
    float z = (neuron->w1*x1)+(neuron->w2*x2)+neuron->bias;
    return relu(z);
}

Neuron neuron_create(){
    Neuron neuron;

    neuron.w1 = ((float)rand()/RAND_MAX)*2.0f-1.0f;
    neuron.w2 = ((float)rand()/RAND_MAX)*2.0f-1.0f;
    neuron.bias = ((float)rand()/RAND_MAX)*2.0f-1.0f;

    return neuron;
}

void neuron_train(Neuron *neuron,float x1,float x2,float target,float learning_rate){
    float output = neuron_forward(neuron,x1,x2); 
    float error = target - output;
    neuron->w1=neuron->w1+learning_rate*error*x1;
    neuron->w2=neuron->w2+learning_rate*error*x2;
    neuron->bias=neuron->bias+learning_rate*error;
}





