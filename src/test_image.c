#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "neural_network.h"
#include "activation.h"
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

    for(int i =0;i<784;i++){
        unsigned char pixel;

        fread(&pixel,1,1,file);

        image[i] = pixel / 255.0f;
    }

    fclose(file);

    return 1;
}

// Find the predicted digit
int predict_digit(Network *network,float *image)
{
    network_forward(network,image);

    int prediction = 0;

    for(int i =1;i<10;i++){
        if(network->activations[network->layer_count][i] >
           network->activations[network->layer_count][prediction]){
            prediction = i;
        }
    }

    return prediction;
}

int main(int argc,char **argv)
{
    if(argc < 2){
        printf("Usage:\n");
        printf("test_image.exe drawings/7_001.pgm\n");
        return 0;
    }

    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading model...\n");

    if(!model_load(&network,"models/model.bin")){
        printf("Could not load model!\n");
        network_free(&network);
        return 0;
    }

    printf("Model loaded successfully!\n\n");

    float image[784] = {0};

    printf("Loading image: %s\n",argv[1]);

    if(!load_pgm(argv[1],image)){
        network_free(&network);
        return 0;
    }

    printf("Image loaded successfully!\n");

    int prediction = predict_digit(&network,image);

    float confidence =
        network.activations[network.layer_count][prediction] * 100.0f;

    printf("\n--- PREDICTION ---\n");
    printf("Predicted: %d\n",prediction);
    printf("Confidence: %.2f%%\n\n",confidence);

    printf("Probabilities:\n");

    for(int i =0;i<10;i++){
        float probability =
            network.activations[network.layer_count][i] * 100.0f;

        printf("%d: %.2f%%\n",i,probability);
    }

    network_free(&network);

    return 0;
}