#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// Get the digit label from the filename
int get_label(char *filename)
{
    return filename[0] - '0';
}

// Calculate loss
float calculate_loss(Network *network,int label)
{
    float probability =
        network->activations[network->layer_count][label];

    if(probability < 1e-7f){
        probability = 1e-7f;
    }

    return -logf(probability);
}

int main()
{
    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading original model...\n");

    if(!model_load(&network,"models/model.bin")){
        printf("Could not load model!\n");
        network_free(&network);
        return 0;
    }

    printf("Original model loaded successfully!\n\n");

    char command[300];

    sprintf(command,"dir /b drawings\\*.pgm");

    FILE *pipe = _popen(command,"r");

    if(pipe == NULL){
        printf("Could not open drawings folder!\n");
        network_free(&network);
        return 0;
    }

    char filenames[500][200];

    int image_count = 0;

    while(fgets(filenames[image_count],
                sizeof(filenames[image_count]),
                pipe))
    {
        filenames[image_count]
            [strcspn(filenames[image_count],"\r\n")] = '\0';

        image_count++;

        if(image_count >= 500){
            break;
        }
    }

    _pclose(pipe);

    printf("Custom images: %d\n\n",image_count);

    if(image_count == 0){
        printf("No images found!\n");
        network_free(&network);
        return 0;
    }

    int epochs = 10;

    float learning_rate = 0.005f;

    printf("--- CUSTOM TRAINING ---\n");
    printf("Epochs: %d\n",epochs);
    printf("Learning rate: %.4f\n\n",learning_rate);

    float image[784];

    for(int epoch =0;epoch<epochs;epoch++)
    {
        float total_loss = 0.0f;

        int correct = 0;

        for(int i =0;i<image_count;i++)
        {
            char path[300];

            sprintf(path,"drawings/%s",filenames[i]);

            if(!load_pgm(path,image)){
                continue;
            }

            int label = get_label(filenames[i]);

            network_forward(&network,image);

            float loss =
                calculate_loss(&network,label);

            total_loss += loss;

            int prediction = 0;

            for(int j =1;j<10;j++){
                if(network.activations[network.layer_count][j] >
                   network.activations[network.layer_count][prediction]){
                    prediction = j;
                }
            }

            if(prediction == label){
                correct++;
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

        float accuracy =
            (float)correct / image_count * 100.0f;

        float average_loss =
            total_loss / image_count;

        printf(
            "Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n",
            epoch + 1,
            average_loss,
            accuracy
        );
    }

    printf("\nSaving custom model...\n");

    if(model_save(&network,"models/custom_model.bin")){
        printf("Custom model saved successfully!\n");
    }else{
        printf("Could not save custom model!\n");
    }

    network_free(&network);

    return 0;
}