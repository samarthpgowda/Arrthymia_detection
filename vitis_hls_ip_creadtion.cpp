#include "mlp.h"
#include "weights.h"

void mlp(data_t input[INPUT_SIZE], int &output)
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t hidden[HIDDEN_SIZE];


// ===============================
// Layer 1 (Hidden Layer)
// ===============================
LAYER1:
    for(int j = 0; j < HIDDEN_SIZE; j++){

        acc_t acc = 0;

    MAC_LOOP:
        for(int i = 0; i < INPUT_SIZE; i++){

            acc_t product =
                ((acc_t)input[i] * (acc_t)W1[i][j]) >> FRAC_BITS;

            acc += product;
        }

        acc += (acc_t)b1[j];

        if(acc < 0)
            acc = 0;

        hidden[j] = (data_t)acc;
    }


// ===============================
// Output Layer
// ===============================
    acc_t acc_out = 0;

OUTPUT_LAYER:
    for(int j = 0; j < HIDDEN_SIZE; j++){

        acc_t product =
            ((acc_t)hidden[j] * (acc_t)W2[j]) >> FRAC_BITS;

        acc_out += product;
    }

    acc_out += (acc_t)b2;


// ===============================
// Final Decision
// ===============================
    output = (acc_out > OUTPUT_THRESHOLD) ? 1 : 0;
}

