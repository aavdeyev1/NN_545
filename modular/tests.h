#pragma once
#include <string>
#include <vector>
#include "tensor.h"
#include "tensor_ops.h"
#include "layer.h"
#include "dataloader.h"

void runAllTests(Device testDevice);

void testAddTensors(Device testDevice);

void testLayerSmoke(Device testDevice);

void testTensorSmoke(Device testDevice);

void testMatmulSmoke(Device testDevice);

void testMatmulCorrect(Device testDevice);

void testTransposedCopyCorrect(Device testDevice);

void testCSVLoader();

void testGetTrainBatch(Device testDevice);