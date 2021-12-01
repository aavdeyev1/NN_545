#pragma once
#include <stdlib.h>
#include <random>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include "tensor.h"

struct TrainBatch
{
private:
    Tensor inputs;
    Tensor targets;
public:
    TrainBatch(TensorShape inputsShape, TensorShape outputsShape);
    TensorShape getInputsShape();
    TensorShape getTargetsShape();
    Tensor getInputs();
    Tensor getTargets();
};

class CSVTrainTestDataLoader
{
private:
    int xNumTrainRows;
    int xNumTestRows;
    int yNumTrainRows;
    int yNumTestRows;
    int xNumFeatures;
    int yNumFeatures;

public:
    std::string xTrainPath, xTestPath, yTrainPath, yTestPath;
    std::unique_ptr<Tensor> xTrain;
    std::unique_ptr<Tensor> xTest;
    std::unique_ptr<Tensor> yTrain;
    std::unique_ptr<Tensor> yTest;

    CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath);
    void loadAll();
    void loadTrainingBatch(TrainBatch &batch);

};

Tensor* CSVRowsToTensor(const std::vector<std::vector<std::string>> &cells);

void loadCSVCells(std::string path, Tensor*  out);

std::vector<int> generateIdxs(int numIdxs, int numRows);

int getNumColsInFirstRowOfFile(std::string path);

int getNumNonBlankLinesInFile(std::string path, int expectedCols);

int numTokens(std::string str);