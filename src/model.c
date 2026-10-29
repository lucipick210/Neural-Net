#include "model.h"
#include <stdio.h>

int model_save(Network *network, const char *filename)
{
    FILE *file = fopen(filename, "wb");

    if(file == NULL)
    {
        printf("Could not save model!\n");
        return 0;
    }

    /*
        Salvam numarul de layere.
    */
    fwrite(&network->layer_count,
           sizeof(int),
           1,
           file);

    /*
        Salvam dimensiunile retelei.
    */
    fwrite(network->sizes,
           sizeof(int),
           network->layer_count + 1,
           file);

    /*
        Salvam fiecare neuron.
    */
    for(int l = 0; l < network->layer_count; l++)
    {
        Layer *layer = &network->layers[l];

        for(int n = 0; n < layer->neuron_count; n++)
        {
            Neuron *neuron = &layer->neurons[n];

            /*
                Bias
            */
            fwrite(&neuron->bias,
                   sizeof(float),
                   1,
                   file);

            /*
                Weights
            */
            fwrite(neuron->weights,
                   sizeof(float),
                   neuron->input_count,
                   file);
        }
    }

    fclose(file);

    printf("Model saved: %s\n", filename);

    return 1;
}


int model_load(Network *network, const char *filename)
{
    FILE *file = fopen(filename, "rb");

    if(file == NULL)
    {
        printf("Could not load model!\n");
        return 0;
    }

    int layer_count;

    fread(&layer_count,
          sizeof(int),
          1,
          file);

    /*
        Verificam daca arhitectura este aceeasi.
    */
    if(layer_count != network->layer_count)
    {
        printf("Model architecture mismatch!\n");
        fclose(file);
        return 0;
    }

    /*
        Citim dimensiunile si verificam.
    */
    int *sizes = network->sizes;

    for(int i = 0; i <= network->layer_count; i++)
    {
        int saved_size;

        fread(&saved_size,
              sizeof(int),
              1,
              file);

        if(saved_size != sizes[i])
        {
            printf("Model size mismatch!\n");
            fclose(file);
            return 0;
        }
    }

    /*
        Citim neuronii.
    */
    for(int l = 0; l < network->layer_count; l++)
    {
        Layer *layer = &network->layers[l];

        for(int n = 0; n < layer->neuron_count; n++)
        {
            Neuron *neuron = &layer->neurons[n];

            /*
                Bias
            */
            fread(&neuron->bias,
                  sizeof(float),
                  1,
                  file);

            /*
                Weights
            */
            fread(neuron->weights,
                  sizeof(float),
                  neuron->input_count,
                  file);
        }
    }

    fclose(file);

    printf("Model loaded: %s\n", filename);

    return 1;
}