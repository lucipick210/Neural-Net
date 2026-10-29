#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    for(int i = 0;i < 784;i++){
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

int main()
{
    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading custom model...\n");

    if(!model_load(&network,"models/custom_model.bin")){
        printf("Could not load custom model!\n");
        network_free(&network);
        return 0;
    }

    printf("Custom model loaded successfully!\n\n");

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

    int confusion[10][10] = {0};

    int correct = 0;

    float image[784];

    printf("--- TESTING CUSTOM MODEL ---\n\n");

    for(int i = 0;i < image_count;i++)
    {
        char path[300];

        sprintf(path,"drawings/%s",filenames[i]);

        if(!load_pgm(path,image)){
            continue;
        }

        int label = get_label(filenames[i]);

        network_forward(&network,image);

        int prediction = 0;

        for(int j = 1;j < 10;j++){
            if(network.activations[network.layer_count][j] >
               network.activations[network.layer_count][prediction]){
                prediction = j;
            }
        }

        float confidence =
            network.activations[network.layer_count][prediction] * 100.0f;

        printf(
            "%s -> Predicted: %d | Confidence: %.2f%%\n",
            filenames[i],
            prediction,
            confidence
        );

        confusion[label][prediction]++;

        if(prediction == label){
            correct++;
        }
    }

    float accuracy =
        (float)correct / image_count * 100.0f;

    printf("\n-----------------------------\n");
    printf("Custom model accuracy: %.2f%%\n",accuracy);
    printf("Correct: %d / %d\n",correct,image_count);
    printf("-----------------------------\n");

    printf("\n--- CONFUSION MATRIX ---\n\n");

    printf("Real \\ Pred    0    1    2    3    4    5    6    7    8    9\n");

    for(int i = 0;i < 10;i++)
    {
        printf("%10d",i);

        for(int j = 0;j < 10;j++)
        {
            printf("%5d",confusion[i][j]);
        }

        printf("\n");
    }

    network_free(&network);

    return 0;
}