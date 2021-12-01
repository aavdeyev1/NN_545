#include "dataloader.h"

TrainBatch::TrainBatch(TensorShape XShape, TensorShape YShape):
idxs(XShape.x),
X(CPU, XShape),
Y(CPU, YShape) {
    if(XShape.x != YShape.x) {
        throw std::runtime_error("X and Y tensors do not match on batch dimension.");
    }

}

Tensor TrainBatch::getX() {
    return X;
}

Tensor TrainBatch::getY() {
    return Y;
}

TensorShape TrainBatch::getXShape(){
    return X.getShape();
}

TensorShape TrainBatch::getYShape(){
    return Y.getShape();
}

CSVTrainTestDataLoader::CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath) :
xTrainPath(xTrainPath),
xTestPath(xTestPath),
yTrainPath(yTrainPath),
yTestPath(yTestPath),
xTrain(nullptr),
yTrain(nullptr),
xTest(nullptr),
yTest(nullptr)
{
    xNumFeatures = getNumColsInFirstRowOfFile(xTrainPath);
    xNumTrainRows = getNumNonBlankLinesInFile(xTrainPath, xNumFeatures);
    xNumTestRows = getNumNonBlankLinesInFile(xTestPath, xNumFeatures);

    yNumFeatures = getNumColsInFirstRowOfFile(yTrainPath);
    yNumTrainRows = getNumNonBlankLinesInFile(yTrainPath, yNumFeatures);
    yNumTestRows = getNumNonBlankLinesInFile(yTestPath, yNumFeatures);

    if(xNumTrainRows != yNumTrainRows){
        std::cout << "ERROR: Need to have the same number of rows in X and Y train CSV files." << std::endl;
        exit(1);
    }
    if(xNumTestRows != yNumTestRows){
        std::cout << "ERROR: Need to have the same number of rows in X and Y test CSV files." << std::endl;
        exit(1);
    }

    xTrain =new Tensor(CPU, {xNumTrainRows, xNumFeatures});
    yTrain =new Tensor(CPU, {yNumTrainRows, yNumFeatures});
    xTest =new Tensor(CPU, {xNumTestRows, xNumFeatures});
    yTest =new Tensor(CPU, {yNumTestRows, yNumFeatures});
}

CSVTrainTestDataLoader::~CSVTrainTestDataLoader(){
    if(xTrain != nullptr) delete xTrain;
    if(yTrain != nullptr) delete yTrain;
    if(xTest != nullptr) delete xTest;
    if(yTest != nullptr) delete yTest;
}

void CSVTrainTestDataLoader::loadAll() {
    loadCSVCells(xTrainPath, xTrain);
    loadCSVCells(yTrainPath, yTrain);
    loadCSVCells(xTestPath, xTest);
    loadCSVCells(yTestPath, yTest);
}

int getNumNonBlankLinesInFile(std::string path, int expectedCols)
{
    std::ifstream in(path);
    std::string line;
    int numLines = 0;
    int lineIdx = 0;
    while ( std::getline(in, line) )
    {
        while(line.length() == 0)
        {
            getline(in, line);
            lineIdx++;
        }
        if(numTokens(line) != expectedCols){
            std::cout << "There was an error: Line " << lineIdx << " does not have the expected number of " << expectedCols << " columns." << std::endl;
        }
        lineIdx++;
        numLines++;
    }
    return numLines;
}

int getNumColsInFirstRowOfFile(std::string path)
{
    int n = 0;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    n = numTokens(line);
    return n;
}

int numTokens(std::string str)
{
    int numTokens = 0;

    int tokenStart = 0;
    int tokenEnd = 0;
    for(int i = 0; i < str.length() - 1; i++) {
        if(str.at(i) == ',') {
            tokenEnd = i - 1;
            if(tokenEnd - tokenStart >= 1)
                numTokens++;

            tokenStart = i + 1;
        }
    }
    tokenEnd = str.length() - 1;
    if(tokenEnd - tokenStart >= 1 && str.at(tokenEnd != ',')) {
        numTokens++;
    }
    return numTokens;
}

void loadCSVCells(std::string path, Tensor * out)
{
    std::ifstream in(path);
    if(!in.is_open()) throw std::runtime_error("Could not open file");

    std::string line;
    std::string token;

    if(out->ndims() != 2) {
        std::cout << "ERROR: CSV file needs to be loaded into a 2D tensor." << std::endl;
        exit(1);
    }

    int tokenIdx = 0;

    while ( std::getline(in, line) )
    {
        std::stringstream ss(line);
        while( std::getline(ss, token, ',')){
            out->data()[tokenIdx] = std::stof(token);
            tokenIdx++;
        }

    }
}

void CSVTrainTestDataLoader::loadTrainingBatch(TrainBatch &batch) {
    // todo: this code is broken.
    int batchSize = batch.getXShape().x;
    int datasetSize = xTrain->getShape().x;
    int numInputFeatures = xTrain->getShape().y;
    int numTargetFeatures = yTrain->getShape().y;

    for(int i = 0; i < batchSize; i++) {
        int idx = std::rand() % datasetSize;
        batch.idxs[i] = idx;

        for(int j = 0; j < numInputFeatures; j++) {
            batch.getX().data()[i * numInputFeatures + j] = xTrain->data()[idx * numInputFeatures + j];
        }
        for(int j = 0; j < numTargetFeatures; j++) {
            batch.getY().data()[i * numTargetFeatures + j] = yTrain->data()[idx * numTargetFeatures + j];
        }
    }
}