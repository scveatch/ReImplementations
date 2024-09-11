#ifndef CNN_H
#define CNN_H

#include "utils.h"

typedef struct {
    // Define CNN properties (like layers, weights, etc.)
} CNN;

typedef struct {
    // Layer configuration (define if needed)
} LayerConfig;

void cnn_init(CNN *cnn, int input_size, int num_layers, LayerConfig *configs);
void cnn_train(CNN *cnn, Data *training_data, Labels *labels);
void cnn_forward(CNN *cnn, Data *input);

#endif
