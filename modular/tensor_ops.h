#pragma once
#include <vector>
#include "matrix_math.h"
#include "tensor.h"

using std::cout;
using std::endl;

void fill(Tensor t, float val);

void addTensors(Tensor A, Tensor B, Tensor out);

void matmul(Tensor A, Tensor B, Tensor out);

void transposedCopy(Tensor in, Tensor out);

void sigmoidFunction(Tensor in, Tensor out, bool forward);

void MSE(Tensor targets, Tensor predictions, Tensor out);

void MSEGradient(Tensor targets, Tensor predictions, Tensor out);