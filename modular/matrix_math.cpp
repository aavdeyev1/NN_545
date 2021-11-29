#include "matrix_math.h"

void matmul2DCPU(float* A, float* B, float *out, int M, int N, int P)
{
    for(int i = 0; i < M; i++){
        for(int j = 0; j < P; j++){
            out[i * P + j] = 0;
            for(int k = 0; k < N; k++){
                out[i * P + j] += A[i * N + k] * B[k *P + j];
            }
        }
    }
}

void transposedCopyMatrix2DCPU(float* in, float* out, int width, int height)
{
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            out[j*width + i] = in[i*width + j];
        }
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void sigmoidFunctionElementwiseCPU(float* in, float* out, int n, bool forward)
{
    for(int i = 0; i < n; i++){
        if(forward){
            out[i] = sigmoid(in[i]);
        } else {
            out[i] = sigmoid(in[i]) * (1 - sigmoid(in[i]));
        }

    }
}