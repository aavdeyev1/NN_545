#pragma once
#include <math.h>
#include "device.h"
#include "tensor.h"

struct ModelMetrics
{
    float mse;
};

float mse(Tensor* real, Tensor* predicted);
float rmse(Tensor * real, Tensor * predicted);