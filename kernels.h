#ifndef KERNELS_H
#define KERNELS_H

void printArray(float *arr, int rows, int cols, int shouldPrint);
__global__ void matrix_multiply_simple(float* a, float* b, float* ab, int m, int n, int k);
__global__ void kernel( int *input, float *output, float *vHidden, float *wHidden, int numIn, int numH, int numLayers, int numPairs );
__device__ float fxGPU(float *x, int idx);
void printArray(int *arr, int rows, int cols, int shouldPrint);
void printArray3D(float *arr, int rows, int cols, int pages, int sP);
// __device__ void add(float *h, float *other);


#endif