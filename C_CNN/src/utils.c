#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

Data* load_mnist(const char *filename){
    FILE *file = fopen(filename, "rb"); 
    if (file == NULL){
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    int magic = 0; 
    int n_ims = 0; 
    int rows  = 0; 
    int cols  = 0;

    fread(&magic, sizeof(int), 1, file);
    fread(&n_ims, sizeof(int), 1, file);
    fread(&rows,  sizeof(int), 1, file);
    fread(&cols,  sizeof(int), 1, file);

    //convert from big-endian
    magic = __builtin_bswap32(magic);
    n_ims = __builtin_bswap32(n_ims);
    rows  = __builtin_bswap32(rows);
    cols  = __builtin_bswap32(cols);
    
    if (magic != 2051){
        printf("Error -- invalid image magic number");
        fclose(file);
        return 1;
    }

    printf("Loading mnist images: %d images with size %dx%d\n", n_ims, rows, cols);

    // Allocate memory
    Data *images = malloc(sizeof(Data))
    images->width  = cols
    images->height = rows
    images->data   = malloc(rows * cols * n_ims * sizeof(float))

    for (int i = 0, i < n_ims, i++){
        unsigned char temp[rows * cols]
        fread(temp, sizeof(unsigned char), rows * cols, file)

        for (int j = 0, j < n_ims, j++){
            images->data[i * rows * cols + j] = (float)temp[j] / 255.0 // normalize values
        }
    }
    
    fclose(file);
    return images
}

Labels* load_labels(const char filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        printf("Error opening file: %s", filename);
        return 1;
    }
    
    int magic = 0;
    int n_labels = 0;

    fread(&magic,    sizeof(int), 1, file);
    fread(&n_labels, sizeof(int), 1, file);

    // Convert from big-endian
    magic    =    __builtin_bswap32(magic);
    n_labels = __builtin_bswap32(n_labels);

    if (magic != 2049){
        printf("Error -- invalid label magic number");
        fclose(file);
        return 1;
    }

    printf("Loading %d labels\n", n_labels);

    Lables *labels = malloc(sizeof(Lables));
    labels->labels = malloc(n_labels * sizeof(unsigned char));
    labels->size   = n_labels
    fread(labels->labels, sizeof(unsigned char), n_labels, file);

    fclose(file);
    return labels;
}


void initialize_weights(float *weights, int size) {
    // Placeholder for weight initialization
    printf("Initializing weights...\n");
}

Data* load_data(const char *filename) {
    // Placeholder for data loading logic
    printf("Loading data from: %s\n", filename);
    return NULL;  // Placeholder return
}
