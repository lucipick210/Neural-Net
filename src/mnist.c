#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned int read_uint32(FILE *file){
    unsigned char bytes[4];

    if(fread(bytes,1,4,file)!=4){
        return 0;
    }
    return((unsigned int)bytes[0]<<24)|((unsigned int)bytes[1]<<16)|((unsigned int)bytes[2]<<8)|((unsigned int)bytes[3]);
}

MNISTDataset mnist_load(const char *image_file,const char *label_file){
    MNISTDataset dataset = {0};

    FILE *images = fopen(image_file,"rb");
    FILE *labels = fopen(label_file,"rb");

    if(images == NULL||labels==NULL){
        printf("Error:couldnot open MNIST files.\n");
        if(images !=NULL){fclose(images);}
        if(labels!=NULL){fclose(labels);}
        return dataset;
    }
    unsigned int image_magic = read_uint32(images);
    unsigned int image_count = read_uint32(images);
    unsigned int rows = read_uint32(images);
    unsigned int cols = read_uint32(images);

    unsigned int label_magic = read_uint32(labels);
    unsigned int label_count = read_uint32(labels);

    if(image_magic!=2051||label_magic!=2049){ 
        printf("Error:invalid MNIST files.\n");
        fclose(images);
        fclose(labels);
        return dataset;
    }
    if(image_count!=label_count){
        printf("Error: image and label counts do not match.\n");
        fclose(images);
        fclose(labels);
        return dataset;
    }
    dataset.count=(int)image_count;
    dataset.rows=(int)rows;
    dataset.cols=(int)cols;

    int pixels = dataset.rows*dataset.cols;

    dataset.images=malloc(dataset.count*pixels*sizeof(float));
    dataset.labels = malloc(dataset.count*sizeof(unsigned char));
    if(dataset.images==NULL||dataset.labels==NULL){
        printf("Error: memory allocation failed\n");
        free(dataset.images);
        free(dataset.labels);
        fclose(images);
        fclose(labels);
        dataset.images = NULL;
        dataset.labels = NULL;
        dataset.count = 0;
        return dataset;
    }
    for(int i =0; i<dataset.count;i++){
        for(int j=0;j<pixels;j++){
            unsigned char pixel;
            if(fread(&pixel,1,1,images)!=1){
                printf("Error: could not read image data\n");
                mnist_free(&dataset);
                fclose(images);
                fclose(labels);
                return dataset;
            }
            dataset.images[i*pixels+j] = pixel/255.0f;
        }
        if(fread(&dataset.labels[i],1,1,labels)!=1){
            printf("Error:could not read label data.\n");
            mnist_free(&dataset);
            fclose(images);
            fclose(labels);
            return dataset;
        }
    }
    fclose(images);
    fclose(labels);
    printf("MNIST loaded succesfully!\n");
    printf("Images:%d\n",dataset.count);
    printf("Size:%dx%d\n",dataset.rows,dataset.cols);
    return dataset;
}

void mnist_free(MNISTDataset *dataset){
    if(dataset == NULL)
        return;
    free(dataset->images);
    free(dataset->labels);

    dataset->images = NULL;
    dataset->labels = NULL;

    dataset->count = 0;
    dataset->cols = 0;
    dataset->rows = 0;
}