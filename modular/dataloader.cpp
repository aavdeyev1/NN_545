#include "dataloader.h"

TrainBatch::TrainBatch():inputs(NULL), targets(NULL) {}

CSVTrainTestDataLoader::CSVTrainTestDataLoader(std::string xTrainPath, std::string xTestPath, std::string yTrainPath, std::string yTestPath) :
xTrainPath(xTrainPath), xTestPath(xTestPath), yTrainPath(yTrainPath), yTestPath(yTestPath) {}

void CSVTrainTestDataLoader::loadAll() {
    std::vector<std::vector<std::string>> cells;

    cells = loadCSVCells(xTrainPath);
    xTrain = CSVRowsToTensor(cells);
    cells.clear();

    cells = loadCSVCells(xTestPath);
    xTest = CSVRowsToTensor(cells);
    cells.clear();

    cells = loadCSVCells(yTrainPath);
    yTrain = CSVRowsToTensor(cells);
    cells.clear();

    cells = loadCSVCells(yTestPath);
    yTest = CSVRowsToTensor(cells);
    cells.clear();
}

Tensor* CSVRowsToTensor(const std::vector<std::vector<std::string>> &cells)
{
    size_t height = cells.size();
    size_t width = cells.at(0).size();
    std::vector<std::string> rowIn;

    Tensor * data = new Tensor(CPU, {height, width});

    for(int i = 0; i < height; i++)
    {
        rowIn = cells.at(i);
        if(rowIn.size() != width){
            std::cout << "ERROR: Malformed data is not a full matrix." << std::endl;
            exit(1);
        }

        for(int j = 0; j < width; j++){
            data->value[i * width + j] = (float) atof(rowIn.at(j).c_str());
        }
    }

    return data;
}

std::vector<std::vector<std::string>> loadCSVCells(std::string path)
{
    std::vector<std::vector<std::string>> cells;
    std::ifstream in(path);
    std::string line;
    while ( std::getline(in, line) )
    {
        cells.push_back(split(line, ','));
    }
    return cells;
}

std::vector<std::string> split (const std::string &s, char delim) {
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
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

void CSVTrainTestDataLoader::loadTrainingBatch(TrainBatch* batch, Device device, int batchSize) {
    int datasetSize = xTrain->getShape().matrixHeight();
    int numInputFeatures = xTrain->getShape().y;
    int numTargetFeatures = yTrain->getShape().y;

    Tensor * inputBatch = new Tensor(device, {batchSize, numInputFeatures});
    Tensor * targetBatch = new Tensor(device, {batchSize, numTargetFeatures});

    std::vector<int> trainIdxs = generateIdxs(batchSize, datasetSize);

    float * inputsSourcePtr;
    float * targetsSourcePtr;

    float * inputsBatchPtr = inputBatch->value;
    float * targetsBatchPtr = targetBatch->value;

    for(int i = 0; i < trainIdxs.size(); i++) {
        int idx = trainIdxs.at(i);
        inputsSourcePtr = &xTrain->value[idx * numInputFeatures];
        targetsSourcePtr = &yTrain->value[idx * numTargetFeatures];
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
    inputBatch->move(device);
    targetBatch->move(device);

    if(batch->inputs != NULL) {
        delete batch->inputs;
    }
    if(batch->targets != NULL){
        delete batch->targets;
    }

    batch->inputs = inputBatch;
    batch->targets = targetBatch;
}