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
#include <string.h>
#include "tensor.h"

struct TrainBatch
{
private:
    Tensor X;
    Tensor Y;
public:
    std::vector<int> idxs;
    TrainBatch(TensorShape XShape, TensorShape YShape);
    TensorShape getXShape();
    TensorShape getYShape();
    Tensor getX();
    Tensor getY();
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
    ~CSVTrainTestDataLoader();
    std::string xTrainPath, xTestPath, yTrainPath, yTestPath;
    Tensor* xTrain;
    Tensor* xTest;
    Tensor* yTrain;
    Tensor* yTest;

    CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath);
    void loadAll();
    void loadTrainingBatch(TrainBatch &batch);

};

Tensor* CSVRowsToTensor(const std::vector<std::vector<std::string>> &cells);

void loadCSVCells(std::string path, Tensor*  out);
int getNumColsInFirstRowOfFile(std::string path);

int getNumNonBlankLinesInFile(std::string path, int expectedCols);

int numTokens(std::string str);