#ifndef KERNELS_H
#define KERNELS_H

void printArray(float *arr, int rows, int cols, int shouldPrint);
__global__ void matrix_multiply_simple(float* a, float* b, float* ab, int m, int n, int k);
__global__ void kernel4( int *input, int *output, int numIn, int numPairs );
__device__ float fxGPU(float x);


#endif