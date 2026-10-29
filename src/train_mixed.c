#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "neural_network.h"
#include "activation.h"
#include "mnist.h"
#include "model.h"

// Load a 28x28 PGM image
int load_pgm(const char *filename,float *image)
{
    FILE *file = fopen(filename,"rb");

    if(file == NULL){
        printf("Could not open image: %s\n",filename);
        return 0;
    }

    char format[3];
    int width;
    int height;
    int max_value;

    fscanf(file,"%2s",&format);
    fscanf(file,"%d %d",&width,&height);
    fscanf(file,"%d",&max_value);

    fgetc(file);

    if(width != 28 || height != 28){
        printf("Image must be 28x28!\n");
        fclose(file);
        return 0;
    }

    for(int i = 0;i < 784;i++)
    {
        unsigned char pixel;

        fread(&pixel,1,1,file);

        image[i] = pixel / 255.0f;
    }

    fclose(file);

    return 1;
}

// Get digit from filename
int get_label(char *filename)
{
    return filename[0] - '0';
}

// Find prediction
int get_prediction(Network *network)
{
    int prediction = 0;

    for(int i = 1;i < 10;i++)
    {
        if(network->activations[network->layer_count][i] >
           network->activations[network->layer_count][prediction])
        {
            prediction = i;
        }
    }

    return prediction;
}

int main()
{
    srand((unsigned int)time(NULL));

    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading original model...\n");

    if(!model_load(&network,"models/model.bin")){
        printf("Could not load original model!\n");
        network_free(&network);
        return 0;
    }

    printf("Original model loaded successfully!\n\n");

    // Load MNIST
    printf("Loading MNIST training dataset...\n");

    MNISTDataset dataset = mnist_load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    );

    if(dataset.images == NULL){
        printf("Could not load MNIST dataset!\n");
        network_free(&network);
        return 0;
    }

    printf("MNIST training dataset loaded!\n");
    printf("Images: %d\n\n",dataset.count);

    // Find custom images
    char command[300];

    sprintf(command,"dir /b drawings\\*.pgm");

    FILE *pipe = _popen(command,"r");

    if(pipe == NULL){
        printf("Could not open drawings folder!\n");

        mnist_free(&dataset);
        network_free(&network);

        return 0;
    }

    char filenames[500][200];

    int custom_count = 0;

    while(fgets(filenames[custom_count],
                sizeof(filenames[custom_count]),
                pipe))
    {
        filenames[custom_count]
            [strcspn(filenames[custom_count],"\r\n")] = '\0';

        custom_count++;

        if(custom_count >= 500){
            break;
        }
    }

    _pclose(pipe);

    printf("Custom images: %d\n\n",custom_count);

    if(custom_count == 0){
        printf("No custom images found!\n");

        mnist_free(&dataset);
        network_free(&network);

        return 0;
    }

    int epochs = 5;

    int mnist_per_epoch = 2000;

    float learning_rate = 0.003f;

    printf("--- MIXED TRAINING ---\n");
    printf("Epochs: %d\n",epochs);
    printf("MNIST images per epoch: %d\n",mnist_per_epoch);
    printf("Custom images per epoch: %d\n",custom_count);
    printf("Learning rate: %.4f\n\n",learning_rate);

    float image[784];

    for(int epoch = 0;epoch < epochs;epoch++)
    {
        int correct_mnist = 0;
        int correct_custom = 0;

        printf("Starting epoch %d...\n",epoch + 1);

        // -------------------------
        // Train on MNIST
        // -------------------------

        for(int i = 0;i < mnist_per_epoch;i++)
        {
            int index = rand() % dataset.count;

            for(int j = 0;j < 784;j++)
            {
                image[j] =
                    dataset.images[index * 784 + j];
            }

            int label = dataset.labels[index];

            network_forward(&network,image);

            int prediction =
                get_prediction(&network);

            if(prediction == label){
                correct_mnist++;
            }

            float targets[10] = {0};

            targets[label] = 1.0f;

            network_backward(
                &network,
                targets,
                learning_rate
            );

            network_update(
                &network,
                learning_rate,
                1
            );
        }

        // -------------------------
        // Train on custom drawings
        // -------------------------

        for(int i = 0;i < custom_count;i++)
        {
            char path[300];

            sprintf(
                path,
                "drawings/%s",
                filenames[i]
            );

            if(!load_pgm(path,image)){
                continue;
            }

            int label =
                get_label(filenames[i]);

            network_forward(&network,image);

            int prediction =
                get_prediction(&network);

            if(prediction == label){
                correct_custom++;
            }

            float targets[10] = {0};

            targets[label] = 1.0f;

            network_backward(
                &network,
                targets,
                learning_rate
            );

            network_update(
                &network,
                learning_rate,
                1
            );
        }

        float mnist_accuracy =
            (float)correct_mnist /
            mnist_per_epoch * 100.0f;

        float custom_accuracy =
            (float)correct_custom /
            custom_count * 100.0f;

        printf(
            "Epoch %d | MNIST: %.2f%% | Custom: %.2f%%\n\n",
            epoch + 1,
            mnist_accuracy,
            custom_accuracy
        );
    }

    // Save mixed model
    printf("Saving mixed model...\n");

    if(model_save(
        &network,
        "models/mixed_model.bin"))
    {
        printf("Mixed model saved successfully!\n");
    }
    else
    {
        printf("Could not save mixed model!\n");
    }

    mnist_free(&dataset);

    network_free(&network);

    return 0;
}