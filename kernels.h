#ifndef KERNELS_H
#define KERNELS_H

__device__ float fxGPU(float *x, int idx);
void printArray(float *arr, int rows, int cols, int shouldPrint);
__global__ void matrix_multiply_simple(float* a, float* b, float* ab, int m, int n, int k);
__global__ void kernel( int *input, float *output, float *vHidden, float *wHidden, float *vOut, float *wOut, float *hError, float *yError, int numIn, int numH, int numOut, int numLayers, int numPairs );

void printArray(int *arr, int rows, int cols, int shouldPrint);
void printArray3D(float *arr, int rows, int cols, int pages, int sP);

#endif