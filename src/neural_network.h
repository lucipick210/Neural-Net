#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct{
    float w1;
    float w2;
    float bias;
}Neuron;
float neuron_forward(Neuron *neuron, float x1,float x2);
Neuron neuron_create();

void neuron_train(Neuron *neuron,
                  float x1,
                  float x2,
                  float target,
                  float learning_rate);

#endif