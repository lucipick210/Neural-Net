#include <stdio.h>

#include "neural_network.h"
#include "mnist.h"
#include "model.h"

int main()
{
    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading mixed model...\n");

    if(!model_load(&network,"models/mixed_model.bin")){
        printf("Could not load mixed model!\n");
        network_free(&network);
        return 0;
    }

    printf("Mixed model loaded successfully!\n\n");

    printf("Loading MNIST test dataset...\n");

    MNISTDataset test_dataset = mnist_load(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );

    if(test_dataset.images == NULL){
        printf("Could not load test dataset!\n");
        network_free(&network);
        return 0;
    }

    printf("Test dataset loaded!\n");
    printf("Images: %d\n",test_dataset.count);
    printf("Size: %dx%d\n\n",
           test_dataset.rows,
           test_dataset.cols);

    int correct = 0;

    printf("--- TESTING MIXED MODEL ON MNIST ---\n\n");

    for(int i = 0;i < test_dataset.count;i++)
    {
        float image[784];

        // Copy normalized MNIST image
        for(int j = 0;j < 784;j++)
        {
            image[j] =
                test_dataset.images[i * 784 + j];
        }

        int label = test_dataset.labels[i];

        network_forward(&network,image);

        int prediction = 0;

        for(int j = 1;j < 10;j++)
        {
            if(network.activations[network.layer_count][j] >
               network.activations[network.layer_count][prediction])
            {
                prediction = j;
            }
        }

        if(prediction == label){
            correct++;
        }

        if((i + 1) % 1000 == 0)
        {
            printf(
                "Tested: %d / %d | Correct: %d\n",
                i + 1,
                test_dataset.count,
                correct
            );
        }
    }

    float accuracy =
        (float)correct /
        test_dataset.count *
        100.0f;

    printf("\n-----------------------------\n");
    printf("Mixed model MNIST results\n");
    printf("-----------------------------\n");

    printf("Correct: %d / %d\n",
           correct,
           test_dataset.count);

    printf("Accuracy: %.2f%%\n",
           accuracy);

    printf("-----------------------------\n");

    mnist_free(&test_dataset);
    network_free(&network);

    return 0;
}