#ifndef LAYERS_H
#define LAYERS_H

#include "utils.h"

typedef struct {
    // Define layer structure (weights, biases, etc.)
} Layer;

void conv_layer_forward(Layer *layer, Data *input);
void pooling_layer_forward(Layer *layer, Data *input);
void fully_connected_layer_forward(Layer *layer, Data *input);

#endif
