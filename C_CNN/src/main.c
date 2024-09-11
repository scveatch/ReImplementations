#include "cnn.h"
#include "utils.h"
#include <stdio.h>

int main() {
    printf("Initializing CNN...\n");
    CNN cnn;
    
    // Example of initializing and training the CNN
    cnn_init(&cnn, 28, 3, NULL);  // Placeholder values
    cnn_train(&cnn, NULL, NULL);  // Placeholder for data and labels
    
    printf("CNN training complete.\n");
    return 0;
}