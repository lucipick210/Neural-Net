#ifndef MODEL_H
#define MODEL_H

#include "neural_network.h"

int model_save(Network *network, const char *filename);
int model_load(Network *network, const char *filename);

#endif