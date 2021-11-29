#pragma once
#include <vector>
#include "matrix_math.h"
#include "tensor.h"

void fill(Tensor* t, float val);

void addTensors(Tensor* A, Tensor* B, Tensor *out);

void matmul(Tensor* A, Tensor* B, Tensor* out);

void transposedCopy(Tensor* in, Tensor* out);

void sigmoidFunction(Tensor* in, Tensor* out, bool forward);

void MSEGradient(Tensor* targets, Tensor* inputs, Tensor* out);

Tensor * matrixFromTensorVector(std::vector<Tensor*> tensors);
