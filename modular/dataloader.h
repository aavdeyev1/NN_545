#pragma once
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "tensor.h"

struct TrainBatch
{
    Tensor* inputs;
    Tensor* targets;
    TrainBatch();
};

class CSVTrainTestDataLoader
{
private:

public:
    std::string xTrainPath, xTestPath, yTrainPath, yTestPath;
    Tensor* xTrain;
    Tensor* xTest;
    Tensor* yTrain;
    Tensor* yTest;
    CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath);
    void loadAll();
    void loadTrainingBatch(TrainBatch* batch, Device device, int batchSize);

};

Tensor* CSVRowsToTensor(const std::vector<std::vector<std::string>> &cells);

std::vector<std::vector<std::string>> loadCSVCells(std::string path);

std::vector<std::string> split (const std::string &s, char delim);

std::vector<int> generateIdxs(int numIdxs, int numRows);