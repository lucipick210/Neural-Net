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

    for(int i =0;i<784;i++){
        unsigned char pixel;

        fread(&pixel,1,1,file);

        image[i] = pixel / 255.0f;
    }

    fclose(file);

    return 1;
}

// Get the predicted digit
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

// Get the label from the filename
int get_label(char *filename)
{
    return filename[0] - '0';
}

// Print the confusion matrix
void print_confusion_matrix(int matrix[10][10])
{
    printf("\n--- CONFUSION MATRIX ---\n\n");

    printf("Real \\ Pred");

    for(int i =0;i<10;i++){
        printf("%5d",i);
    }

    printf("\n");

    for(int real =0;real<10;real++){

        printf("%10d",real);

        for(int predicted =0;predicted<10;predicted++){
            printf("%5d",matrix[real][predicted]);
        }

        printf("\n");
    }
}

int main()
{
    int architecture[] = {784,64,32,10};

    Network network = network_create(architecture,3);

    printf("Loading model...\n");

    if(!model_load(&network,"models/model.bin")){
        printf("Could not load model!\n");
        network_free(&network);
        return 0;
    }

    printf("Model loaded successfully!\n\n");

    char command[300];

    sprintf(command,"dir /b drawings\\*.pgm");

    FILE *pipe = _popen(command,"r");

    if(pipe == NULL){
        printf("Could not open drawings folder!\n");
        network_free(&network);
        return 0;
    }

    char filename[200];

    int total = 0;
    int correct = 0;

    float image[784];

    int confusion_matrix[10][10] = {0};

    printf("--- CUSTOM DATASET TEST ---\n\n");

    while(fgets(filename,sizeof(filename),pipe))
    {
        filename[strcspn(filename,"\r\n")] = '\0';

        char path[300];

        sprintf(path,"drawings/%s",filename);

        if(!load_pgm(path,image)){
            continue;
        }

        int label = get_label(filename);

        int prediction = predict_digit(&network,image);

        confusion_matrix[label][prediction]++;

        if(prediction == label){

            printf("%s -> Predicted: %d  OK\n",
                   filename,prediction);

            correct++;

        }else{

            printf("%s -> Predicted: %d  WRONG (expected %d)\n",
                   filename,prediction,label);
        }

        total++;
    }

    _pclose(pipe);

    printf("\n--- RESULTS ---\n");

    printf("Images: %d\n",total);
    printf("Correct: %d\n",correct);
    printf("Wrong: %d\n",total-correct);

    if(total > 0){

        float accuracy =
            (float)correct / total * 100.0f;

        printf("Accuracy: %.2f%%\n",accuracy);
    }

    print_confusion_matrix(confusion_matrix);

    network_free(&network);

    return 0;
}