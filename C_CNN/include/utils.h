#ifndef UTILS_H
#define UTILS_H

// Struct representing image data
typedef struct {
    int width;
    int height;
    float *data;  // Store pixel as float
} Data;

// Struct for vector of image labels
typedef struct {
    unsigned char *labels;  // Array of labels
    int size;     // Number of labels
} Labels;

void initialize_weights(float *weights, int size);
Data* load_data(const char *filename);

#endif
