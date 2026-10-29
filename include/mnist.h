#ifndef MNIST_H
#define MNIST_H

typedef struct{
    float *images;
    unsigned char *labels;

    int count;
    int rows;
    int cols;
}MNISTDataset;

MNISTDataset mnist_load(const char *image_file,const char *label_file);

void mnist_free(MNISTDataset *dataset);

#endif