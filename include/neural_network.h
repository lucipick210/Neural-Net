#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct{
    float *weights;
    float *weight_gradients;
    float bias;
    float bias_gradients;
    int input_count;
}Neuron;

typedef struct{
    Neuron *neurons;
    int neuron_count;
    int input_count;
}Layer;

Neuron neuron_create(int input_count);
void neuron_free(Neuron *neuron);
float neuron_forward(Neuron *neuron, float *inputs);

Layer layer_create(int neuron_count,int input_count);
void layer_free(Layer *layer);
void layer_forward(Layer *layer, float *inputs, float *outputs,int use_activation);
void layer_backward(Layer *layer, float *inputs,float *neuron_gradients,float *input_gradients,float learning_rate,int use_activation);

typedef struct{
    Layer *layers;
    int layer_count;
    int *sizes;
    float **activations;
    float **gradients;
}Network;

Network network_create(int *architecture,int layer_count);

void network_free(Network *network);
void network_forward(Network *network,float *inputs);
void network_backward(Network *network, float *targets,float learning_rate);
void network_update(Network *network,float learning_rate,int batch_size);

#endif