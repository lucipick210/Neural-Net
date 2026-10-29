#include "neural_network.h"
#include "activation.h"
#include<stdlib.h>

Neuron neuron_create(int input_count){
    Neuron neuron;
    neuron.input_count = input_count;
    neuron.weights = malloc(input_count*sizeof(float));
    neuron.weight_gradients = malloc(input_count*sizeof(float));
    for(int i = 0; i<input_count;i++){
        neuron.weights[i]=((float)rand()/RAND_MAX)*0.1f-0.05f;
        neuron.weight_gradients[i]=0.0f;
    }
    neuron.bias=0.0f;
    neuron.bias_gradients = 0.0f;
    return neuron;
}

float neuron_forward(Neuron *neuron,float *inputs){
    float z = neuron->bias;
    for(int i=0;i<neuron->input_count;i++){
        z+=neuron->weights[i]*inputs[i];
    }
    return leaky_relu(z);
}

Layer layer_create(int neuron_count,int input_count){
    Layer layer;
    layer.neuron_count=neuron_count;
    layer.input_count=input_count;
    layer.neurons=malloc(neuron_count * sizeof(Neuron));
    for(int i=0;i<neuron_count;i++){
        layer.neurons[i]=neuron_create(input_count);
    }
    return layer;
}

void layer_forward(Layer *layer,float *inputs,float *outputs,int use_activation){
    for(int i=0;i<layer->neuron_count;i++){
        float z = layer->neurons[i].bias;
        for(int j =0; j<layer->input_count;j++){
            z+=layer->neurons[i].weights[j]*inputs[j];
        }
        if(use_activation){
            outputs[i] = leaky_relu(z);
        }else{
            outputs[i]= z;
        }
    }
}


void layer_free(Layer *layer){
    if(layer == NULL)
        return;
    for(int i =0; i<layer->neuron_count;i++){
        free(layer->neurons[i].weights);
        free(layer->neurons[i].weight_gradients);
    }
    free(layer->neurons);
    layer->neurons = NULL;
    layer->neuron_count =0;
    layer->input_count =0;
}

void layer_backward(Layer *layer,float *inputs,float *neuron_gradients,float *input_gradients,float learning_rate,int use_activation){
    for(int i=0; i< layer->input_count;i++){
        input_gradients[i] = 0.0f;
    }
    for(int i=0;i<layer->neuron_count;i++){
        Neuron *neuron = &layer->neurons[i];
        float z = neuron->bias;
        
        for(int j =0; j< neuron->input_count;j++){
            z+=neuron->weights[j]*inputs[j];
        }

        float gradient_z;
        if (use_activation){
            float activation_gradient;
            if(z>0.0f)
                activation_gradient = 1.0f;
            else
                activation_gradient = 0.01f;
            gradient_z = neuron_gradients[i]*activation_gradient;
        }else{
            gradient_z = neuron_gradients[i];
        }
        
        for(int j =0; j< neuron->input_count;j++){
            input_gradients[j] += gradient_z*neuron->weights[j];
        }
        
        for(int j =0; j < neuron->input_count;j++){
            float gradient_weight = gradient_z*inputs[j];
            neuron->weight_gradients[j] +=gradient_weight;
        }
        neuron->bias_gradients += gradient_z;
    }

}

Network network_create(int *architecture, int layer_count){
    Network network;

    network.layer_count = layer_count;

    network.sizes = malloc((layer_count+1)*sizeof(int));

    for(int i = 0; i<= layer_count;i++){
       network.sizes[i]=architecture[i];
    }
    network.layers = malloc(layer_count*sizeof(Layer));

    for(int i = 0;i<layer_count;i++){
        network.layers[i] = layer_create(architecture[i+1],architecture[i]);
    }
    network.activations = malloc((layer_count+1)*sizeof(float*));
    network.gradients = malloc((layer_count+1)*sizeof(float*));

    for(int i = 0; i<=layer_count;i++){
        network.activations[i] = malloc(network.sizes[i]*sizeof(float));
        network.gradients[i] = malloc(network.sizes[i]*sizeof(float));
    }
    return network;
}

void network_free(Network *network){
    if(network==NULL)
        return;
    for(int i=0;i<network->layer_count;i++){
        layer_free(&network->layers[i]);
    }
    free(network->layers);
    for(int i = 0; i<=network->layer_count;i++){
        free(network->activations[i]);
        free(network->gradients[i]);
    }
    free(network->activations);
    free(network->gradients);
    free(network->sizes);
    network->layers = NULL;
    network->activations = NULL;
    network->gradients=NULL;
    network->sizes = NULL;
    network->layer_count = 0;
}

void network_forward(Network *network,float *inputs){

    for(int i =0; i< network->sizes[0];i++){
        network->activations[0][i] = inputs[i];
    }
        for (int i = 0; i < network->layer_count-1; i++)
    {
        layer_forward(&network->layers[i],network->activations[i],network->activations[i + 1],1);
    }
    int last = network->layer_count-1;

    layer_forward(&network->layers[last],network->activations[last],network->activations[last+1],0);
    softmax(network->activations[last+1],network->activations[last+1],network->sizes[network->layer_count]);


}


void network_backward(Network *network,float *targets,float learning_rate){
    int last_layer = network->layer_count;
    int output_count=network->sizes[last_layer];

    for(int i =0;i<output_count;i++){
        float output = network->activations[last_layer][i];
        network->gradients[last_layer][i] = output-targets[i];
    }

    for(int i = last_layer-1; i>=0; i--){
        int use_activation = (i!=last_layer-1);
        layer_backward(&network->layers[i],network->activations[i],network->gradients[i+1],network->gradients[i],learning_rate,use_activation);

    }
}

void network_update(Network *network,float learning_rate,int batch_size){
    for(int layer =0;layer<network->layer_count;layer++){
        Layer *current_layer = &network->layers[layer];
        for(int neuron=0; neuron<current_layer->neuron_count;neuron++){
            Neuron *current_neuron = &current_layer->neurons[neuron];
            for(int weight =0; weight < current_neuron->input_count;weight++){
                float average_gradient = current_neuron->weight_gradients[weight]/batch_size;
                current_neuron->weights[weight]-=learning_rate*average_gradient;
                current_neuron->weight_gradients[weight] =0.0f;
            }
            float average_bias_gradient = current_neuron->bias_gradients/batch_size;
            current_neuron->bias-=learning_rate*average_bias_gradient;
            current_neuron->bias_gradients =0.0f;
        }
    }
}