#include "neural_network.h"
#include "activation.h"
#include <stdio.h>
#include <math.h>

float calculate_loss(Network *network, float *input, float *targets)
{
    network_forward(network, input);

    int output_layer = network->layer_count;

    float loss = 0.0f;

    for(int i = 0; i < network->sizes[output_layer]; i++){
        float output = network->activations[output_layer][i];

        if(output < 1e-7f)
            output = 1e-7f;

        loss -= targets[i] * logf(output);
    }

    return loss;
}

float relative_error(float a,float b){
    float numerator = fabsf(a-b);
    float denominator = fabsf(a) + fabsf(b) + 1e-8f;
    return numerator / denominator;
}

float check_weight(Network *network,float *input,float *targets,int layer,int neuron,int weight_index,int *failed){
    float *weight = &network->layers[layer].neurons[neuron].weights[weight_index];

    float original_weight = *weight;
    float epsilon = 0.001f;

    *weight = original_weight + epsilon;

    float loss_plus = calculate_loss(network,input,targets);
    *weight = original_weight - epsilon;

    float loss_minus = calculate_loss(network,input,targets);
    *weight = original_weight;

    float numerical_gradient = (loss_plus-loss_minus)/(2.0f*epsilon);

    network_forward(network,input);
    network_backward(network,targets,0.0f);
    float gradient_z;

    if(layer == network->layer_count - 1){
        gradient_z = network->gradients[layer+1][neuron];
    }else{
        float z = network->layers[layer].neurons[neuron].bias;
        for(int i= 0; i<network->layers[layer].input_count;i++){
            z+=network->layers[layer].neurons[neuron].weights[i]*network->activations[layer][i];
        }
        printf("  z=%.10f\n",z);
        float activation_gradient;
        if(z>0.0f)
            activation_gradient = 1.0f;
        else
            activation_gradient = 0.01f;
        gradient_z = network->gradients[layer+1][neuron]*activation_gradient;
    }
    float backprop_gradient = gradient_z*network->activations[layer][weight_index];
    float difference = fabsf(backprop_gradient-numerical_gradient);
    float relative = relative_error(backprop_gradient,numerical_gradient);
    int passed = (difference<0.001f || relative < 0.01f);
    printf("W|Layer:%d Neuron:%d Weight:%d|"
            "Back:%.8f Num:%.8f Diff:%.8f Rel:%.6f [%s]\n",
            layer,neuron,weight_index,
            backprop_gradient,numerical_gradient,difference,relative,passed? "PASS":"FAIL");
    if(!passed)
        (*failed)++;
    return difference;
}

float check_bias(Network *network,float *input,float *targets,int layer,int neuron,int *failed){
    float *bias = &network->layers[layer].neurons[neuron].bias;

    float original_bias = *bias;

    float epsilon = 0.001f;

    *bias = original_bias+epsilon;
    float loss_plus = calculate_loss(network,input,targets);

    *bias = original_bias-epsilon;
    float loss_minus = calculate_loss(network,input,targets);

    *bias = original_bias;

    float numerical_gradient = (loss_plus-loss_minus) / (2.0f * epsilon);

    network_forward(network,input);
    network_backward(network,targets,0.0f);

    float gradient_z;

    if(layer==network->layer_count-1){
        gradient_z = network->gradients[layer+1][neuron];
    }else{
        float z = network->layers[layer].neurons[neuron].bias;

        for(int i =0;i<network->layers[layer].input_count;i++){
            z+=network->layers[layer].neurons[neuron].weights[i]*network->activations[layer][i];
        }
        float activation_gradient;
        if(z>0.0f)
            activation_gradient = 1.0f;
        else
            activation_gradient = 0.01f;
        gradient_z = network->gradients[layer+1][neuron]*activation_gradient;
    }
    float backprop_gradient = gradient_z;
    float difference = fabsf(backprop_gradient-numerical_gradient);
    float relative = relative_error(backprop_gradient,numerical_gradient);
    int passed = (difference<0.001f||relative<0.01f);
    printf("B | Layer:%d Neuron:%d |"
           "Back:%.8f Num:%.8f Diff:%.8f Rel:%.6f [%s]\n ",
            layer,neuron,backprop_gradient,numerical_gradient,difference,relative,passed ? "PASS":"FAIL");
    if(!passed)
        (*failed)++;
    return difference;
}

int main(void)
{
    int architecture[] = {2, 3, 2};

    Network network = network_create(architecture, 2);

    float input[2] = {0.5f, 0.8f};

    float targets[2] = {1.0f, 0.0f};

    float max_difference = 0.0f;
    int weight_count = 0;
    int bias_count = 0;
    int failed = 0;
    printf("\n--- AUTOMATIC GRADIENT CHECK ---\n\n");
    for(int layer =0;layer<network.layer_count;layer++){
        Layer *current_layer = &network.layers[layer];
        for(int neuron = 0;neuron<current_layer->neuron_count;neuron++){
            for(int weight=0;weight<current_layer->input_count;weight++){
                float difference = check_weight(&network,input,targets,layer,neuron,weight,&failed);
                if(difference>max_difference)
                    max_difference = difference;
                weight_count++;
            }
            float difference = check_bias(&network,input,targets,layer,neuron,&failed);
            if(difference>max_difference)
                max_difference = difference;
            bias_count++;
        }
    }
    printf("\n--- RESULT ---\n\n");

    printf("Weights checked:%d\n",weight_count);

    printf("Biases checked:%d\n",bias_count);
    printf("Maximum difference: %.10f\n",max_difference);

    if(failed == 0){
        printf("\nGRADIENT CHECK PASSED!\n");
    }else{
        printf("\nGRADIENT CHECK FAILED!\n");
        printf("Failed parameters:%d\n",failed);
    }
    network_free(&network);      
    return 0;
}