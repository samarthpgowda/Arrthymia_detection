#include <iostream>
#include <stdio.h>
#include "mlp.h"

#define NUM_SAMPLES 15515   //Change to match your dataset

int main()
{
    data_t input[INPUT_SIZE];
    int output;
    int label;

    int correct = 0;
    int total = 0;

    FILE *input_fp = fopen("X_test.txt","r");
    if(input_fp == NULL){
        std::cout << "ERROR: Cannot open all_inputs.txt\n";
        return 1;
    }

    FILE *label_fp = fopen("y_test.txt","r");
    if(label_fp == NULL){
        std::cout << "ERROR: Cannot open all_labels.txt\n";
        return 1;
    }

    for(int sample = 0; sample < NUM_SAMPLES; sample++)
    {
        // Load one input sample (241 integers)
        for(int i = 0; i < INPUT_SIZE; i++)
        {
            if(fscanf(input_fp,"%hd",&input[i]) != 1)
            {
                std::cout << "ERROR reading input file\n";
                return 1;
            }
        }

        // Load label
        if(fscanf(label_fp,"%d",&label) != 1)
        {
            std::cout << "ERROR reading label file\n";
            return 1;
        }

        // Run accelerator
        mlp(input, output);

        if(output == label)
            correct++;

        total++;
    }

    fclose(input_fp);
    fclose(label_fp);

    float accuracy = (float)correct / total;

    std::cout << "\n===== FULL DATASET TEST =====\n";
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Correct predictions: " << correct << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}

