#pragma once
#include <math.h>

float sigmoid(float x);

void matmul2DCPU(float* A, float* B, float *out, int M, int N, int P);

void transposedCopyMatrix2DCPU(float* in, float* out, int width, int height);

void sigmoidFunctionElementwiseCPU(float* in, float* out, int n, bool forward);