#include "dataloader.h"

TrainBatch::TrainBatch(TensorShape inputsShape, TensorShape targetsShape):
inputs(CPU, inputsShape),
targets(CPU, targetsShape) {}

Tensor TrainBatch::getInputs() {
    return inputs;
}

Tensor TrainBatch::getTargets() {
    return targets;
}

TensorShape TrainBatch::getInputsShape(){
    return inputs.getShape();
}

TensorShape TrainBatch::getTargetsShape(){
    return targets.getShape();
}

CSVTrainTestDataLoader::CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath) :
xTrainPath(xTrainPath),
xTestPath(xTestPath),
yTrainPath(yTrainPath),
yTestPath(yTestPath) {
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

    xTrain.reset(new Tensor(CPU, {xNumTrainRows, xNumFeatures}));
    yTrain.reset(new Tensor(CPU, {yNumTrainRows, yNumFeatures}));
    xTest.reset(new Tensor(CPU, {xNumTestRows, xNumFeatures}));
    yTest.reset(new Tensor(CPU, {yNumTestRows, yNumFeatures}));
}

void CSVTrainTestDataLoader::loadAll() {
    loadCSVCells(xTrainPath, xTrain.get());
    loadCSVCells(yTrainPath, yTrain.get());
    loadCSVCells(xTestPath, xTest.get());
    loadCSVCells(yTestPath, yTest.get());
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
    float parsedToken;
    std::ifstream in(path);
    std::string line;

    if(out->ndims() != 2) {
        std::cout << "ERROR: CSV file needs to be loaded into a 2D tensor." << std::endl;
        exit(1);
    }

    int tokenIdx = 0;

    int nRows = out->getShape().x;
    int nCols = out->getShape().y;

    int col = 0;
    int row = 0;
    while ( std::getline(in, line) )
    {
        col = 0;
        while(line.length() == 0)
        {
            getline(in, line);
        }
        int tokenStart = 0;
        int tokenEnd = 0;
        for(int i = 0; i < line.length() - 1; i++) {
            if(line.at(i) == ',') {
                tokenEnd = i - 1;
                if(tokenEnd - tokenStart >= 1) {
                    parsedToken = std::stof(line.substr(tokenStart, tokenEnd - tokenStart));
                    if(tokenIdx < out->dataLength()) {
                        out->data()[tokenIdx] = parsedToken;
                    } else {
                        std::cout << "ERROR: Too many tokens in input file while loading CSV." << std::endl;
                        exit(1);
                    }
                    tokenIdx += 1;
                }
                tokenStart = i + 1;
            }
        }
        tokenEnd = line.length() - 1;
        if(tokenEnd - tokenStart >= 1 && line.at(tokenEnd != ',')) {
            out->data()[tokenIdx] = std::stof(line.substr(tokenStart, tokenEnd - tokenStart));
        }

        row++;
    }
}

std::vector<int> generateIdxs(int numIdxs, int numRows)
{
    //https://stackoverflow.com/questions/21516575/fill-a-vector-with-random-numbers-c
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {0, numRows};

    auto gen = [&dist, &mersenne_engine](){
        return dist(mersenne_engine);
    };

    std::vector<int> vec(numIdxs);
    std::generate(begin(vec), end(vec), gen);
    return vec;
}

void CSVTrainTestDataLoader::loadTrainingBatch(TrainBatch &batch) {
    // todo: this code is broken.
    int batchSize = batch.getInputsShape().x;
    int datasetSize = xTrain->getShape().matrixHeight();
    int numInputFeatures = xTrain->getShape().y;
    int numTargetFeatures = yTrain->getShape().y;

    std::vector<int> trainIdxs = generateIdxs(batchSize, datasetSize);

    float * inputsSourcePtr;
    float * targetsSourcePtr;

    float * inputsBatchPtr = batch.getInputs().data();
    float * targetsBatchPtr = batch.getTargets().data();

    for(int i = 0; i < trainIdxs.size(); i++) {
        // int idx = trainIdxs.at(i); // todo fixme
        int idx = i;
        inputsSourcePtr = &xTrain->data()[idx * numInputFeatures];
        targetsSourcePtr = &yTrain->data()[idx * numTargetFeatures];
        for(int j = 0; j < numInputFeatures; j++) {
            *inputsBatchPtr = *inputsSourcePtr;
            inputsBatchPtr++;
            inputsSourcePtr++;
        }
        for(int j = 0; j < numTargetFeatures; j++) {
            *targetsBatchPtr = *targetsSourcePtr;
            targetsBatchPtr++;
            targetsSourcePtr++;
        }
    }
}