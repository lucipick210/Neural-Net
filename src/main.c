#include "gui.h"
#include <stdio.h>
#include "mnist.h"
#include "neural_network.h"
#include "activation.h"
#include <math.h>
#include "model.h"

int main()
{
    int architecture[] = {784, 64, 32, 10};
    float image[IMAGE_SIZE] = {0};
    Network network = network_create(architecture, 3);

    FILE *model_file = fopen("models/model.bin", "rb");

    if(model_file != NULL)
    {
        fclose(model_file);

        printf("Model found!\n");
        printf("Loading model...\n");

        if(model_load(&network, "models/model.bin"))
        {
            printf("Model loaded successfully!\n");
            printf("\n--- LOADED MODEL CHECK ---\n");

            for(int l = 0; l < network.layer_count; l++)
            {
                Layer *layer = &network.layers[l];

                printf("Layer %d:\n", l);

                for(int n = 0; n < layer->neuron_count; n++)
                {
                    Neuron *neuron = &layer->neurons[n];

                    if(!isfinite(neuron->bias))
                    {
                        printf("BAD BIAS: layer=%d neuron=%d value=%f\n",
                            l, n, neuron->bias);
                    }

                    for(int w = 0; w < neuron->input_count; w++)
                    {
                        if(!isfinite(neuron->weights[w]))
                        {
                            printf("BAD WEIGHT: layer=%d neuron=%d weight=%d value=%f\n",
                                l, n, w, neuron->weights[w]);
                        }
                    }
                }
            }
        }
        else
        {
            printf("Model loading failed!\n");
            network_free(&network);
            return 1;
        }
    }
    else
    {
        printf("No saved model found.\n");
        printf("Loading MNIST...\n");

        MNISTDataset dataset = mnist_load(
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte");

        if(dataset.images == NULL || dataset.labels == NULL)
        {
            printf("Failed to load MNIST.\n");
            network_free(&network);
            return 1;
        }

        printf("MNIST loaded successfully!\n");

        float targets[10];
        float learning_rate = 0.03f;
        int batch_size = 32;

        for(int epoch = 0; epoch < 5; epoch++)
        {
            float total_loss = 0.0f;

            for(int i = 0; i < 60000; i++)
            {
                float *train_image = &dataset.images[i * 784];

                for(int j = 0; j < 10; j++)
                    targets[j] = 0.0f;

                int label = dataset.labels[i];
                targets[label] = 1.0f;

                network_forward(&network, train_image);

                float output =
                    network.activations[network.layer_count][label];

                if(output < 1e-7f)
                    output = 1e-7f;

                total_loss += -logf(output);

                network_backward(&network, targets, 0.0f);

                if((i + 1) % batch_size == 0)
                {
                    network_update(
                        &network,
                        learning_rate,
                        batch_size);

                    for(int l = 0; l < network.layer_count; l++)
                    {
                        for(int n = 0; n < network.layers[l].neuron_count; n++)
                        {
                            Neuron *neuron =
                                &network.layers[l].neurons[n];

                            if(!isfinite(neuron->weights[0]) ||
                               !isfinite(neuron->bias))
                            {
                                printf("\n!!! NAN APPEARED !!!\n");
                                printf("Epoch: %d\n", epoch + 1);
                                printf("Image: %d\n", i);
                                printf("Layer: %d\n", l);
                                printf("Neuron: %d\n", n);
                                printf("Weight[0]: %f\n",
                                    neuron->weights[0]);
                                printf("Bias: %f\n",
                                    neuron->bias);

                                mnist_free(&dataset);
                                network_free(&network);
                                return 1;
                            }
                        }
                    }
                }
            }

            printf("Epoch %d | Loss: %.6f\n",
                epoch + 1,
                total_loss / 60000.0f);
        }

        printf("Training finished!\n");

        model_save(
            &network,
            "models/model.bin");

        printf("Saving done\n");

        mnist_free(&dataset);
    }

    printf("\nLoading test dataset...\n");

    MNISTDataset test_dataset = mnist_load(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte");

    if(test_dataset.images == NULL ||
       test_dataset.labels == NULL)
    {
        printf("Failed to load test dataset.\n");
        network_free(&network);
        return 1;
    }

    printf("Test dataset loaded!\n");
    printf("\n--- TESTING MODEL ---\n");

    int correct = 0;

    for(int i = 0; i < 10000; i++)
    {
        float *test_image = &test_dataset.images[i * 784];

        network_forward(&network, test_image);

        int predicted = 0;

        for(int j = 1; j < 10; j++)
        {
            if(network.activations[network.layer_count][j] >
               network.activations[network.layer_count][predicted])
            {
                predicted = j;
            }
        }

        if(predicted == test_dataset.labels[i])
            correct++;
    }

    printf("Correct: %d / 10000\n", correct);
    printf("Accuracy: %.2f%%\n",
        (float)correct / 10000.0f * 100.0f);

    mnist_free(&test_dataset);

    printf("Starting Digit Recognizer...\n");

    gui_run(
        image,
        &network);

    network_free(&network);

    printf("Finish\n");

    return 0;
}
