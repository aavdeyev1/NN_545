#include "tests.h"

void runAllTests(Device testDevice)
{
    testAddTensors(testDevice);
    testTensorSmoke(testDevice);
    testLayerSmoke(testDevice);
    testMatmulSmoke(testDevice);
    testMatmulCorrect(testDevice);
    testTransposedCopyCorrect(testDevice);
    testCSVLoader();
    testGetTrainBatch();
    testNumTokens();
}

void testAddTensors(Device testDevice)
{
    if(testDevice == CPU) {
        TensorShape shape = {4};
        Tensor A(CPU, shape);
        fill(A, 1.0f);


        Tensor B(CPU, shape);
        fill(B, 2.0f);

        Tensor C(CPU, shape);

        addTensors(A, B, C);

        for (int i = 0; i < C.dataLength(); i++) {
            if (C.data()[i] != 3.0f) {
                std::cout << "testAddTensors failed." << std::endl;
                exit(1);
            }
        }
    } else if(testDevice == GPU){
        // todo support gpu
    }
    std::cout << "testAddTensors passed." << std::endl;
}

void testTensorSmoke(Device testDevice)
{
    if(testDevice == CPU){
        TensorShape shape = {3, 4};
        Tensor ATensor(CPU, shape);
    } else if(testDevice == GPU){
        // todo support GPU
    }

    std::cout << "testTensorSmoke passed." << std::endl;
}

void testLayerSmoke(Device testDevice)
{
    if(testDevice == CPU) {
        TensorShape inputLayerShape = {4, 4};
        TensorShape linearLayerShape = {4, 8};

        InputLayer inputLayer("testInput", CPU, inputLayerShape);

        LinearLayer linearLayer("testLinear", &inputLayer, 8);

        if (!linearLayer.getOutputShape().equals(linearLayerShape)) {
            std::cout << "testLayerSmoke failed shape test." << std::endl;
            exit(1);
        }
    } else if(testDevice == GPU) {
        // todo support GPU
    }
    std::cout << "testLayerSmoke passed." << std::endl;
}

void testMatmulSmoke(Device testDevice)
{
    if(testDevice == CPU){
        TensorShape shapeA = {3, 4};
        TensorShape shapeB = {4, 7};
        TensorShape shapeC = {3, 7};
        Tensor A(CPU, shapeA);
        fill(A, 1.0f);


        Tensor B(CPU, shapeB);
        fill(B, 2.0f);
        Tensor C(CPU, shapeC);

        matmul(A, B, C);
    }
    else if(testDevice == GPU){
        // todo support gpu
    }
    std::cout << "testMatmulSmoke passed." << std::endl;
}

void testMatmulCorrect(Device testDevice)
{

    TensorShape shape = {2, 2};

    if(testDevice == CPU){
        Tensor A(CPU, shape);
        A.data()[0] = 1;
        A.data()[1] = 2;
        A.data()[2] = 3;
        A.data()[3] = 4;

        Tensor B(CPU, shape);
        B.data()[0] = -4;
        B.data()[1] = -3;
        B.data()[2] = -2;
        B.data()[3] = -1;

        Tensor C(CPU, shape);

        matmul(A, B, C);
        if( C.data()[0] != -8 ||
            C.data()[1] != -5 ||
            C.data()[2] != -20 ||
            C.data()[3] != -13){
            std::cout << "testMatmulCorrect failed first test. Got value:" <<std::endl;
            std::cout << C.toString() << std::endl;
            exit(1);
        }
    } else if (testDevice == GPU){
        // todo support GPU
    }
    std::cout << "testMatmulCorrect passed." << std::endl;
}

void testTransposedCopyCorrect(Device testDevice)
{
    if(testDevice == CPU){
        TensorShape shapeA = {2, 3};
        TensorShape shapeB = {3, 2};
        Tensor A(CPU, shapeA);
        Tensor B(CPU, shapeB);
        A.data()[0] = 1;
        A.data()[1] = 2;
        A.data()[2] = 3;
        A.data()[3] = -1;
        A.data()[4] = -2;
        A.data()[5] = -3;
        transposedCopy(A, B);
        if( B.data()[0] != 1 &&
            B.data()[1] != -1 &&
            B.data()[2] != 2 &&
            B.data()[3] != -2 &&
            B.data()[4] != 3 &&
            B.data()[5] != -3
                ){
            std::cout << "testTransposedCopyCorrect failed, value of out matrix:" << std::endl;
            std::cout << B.toString() << std::endl;
            exit(1);
        }
    } else if(testDevice == GPU){
        // todo support GPU
    }

    std::cout << "testTransposedCopyCorrect passed." << std::endl;
}

void testCSVLoader()
{
    std::string path = "datasets/california_housing_prepared/x_test.csv";
    Tensor* dataCache = new Tensor(CPU, {6192, 8}); // # lines in CSV file, $ columns
    loadCSVCells(path, dataCache);
    /*
    std::cout << "CSV loader test:";

    for(int i = 0; i < cells.size(); i++){
        std::cout << std::endl;
        std::vector<std::string> line = cells.at(i);
        for(int j = 0; j < line.size(); j++){
            std::cout << line.at(j) << ",";
        }
    }
    std::cout << std::endl;
     */
    delete dataCache;
}

void testGetTrainBatch()
{
    int numFeaturesIn = 8;
    int numFeaturesOut = 1;
    int batchSize = 12;
    TrainBatch batch({batchSize, numFeaturesIn}, {batchSize, numFeaturesOut});
    std::string xTrainPath = "datasets/california_housing_prepared/x_train.csv";
    std::string xTestPath = "datasets/california_housing_prepared/x_test.csv";
    std::string yTrainPath = "datasets/california_housing_prepared/y_train.csv";
    std::string yTestPath = "datasets/california_housing_prepared/y_test.csv";

    CSVTrainTestDataLoader loader(xTrainPath, xTestPath, yTrainPath, yTestPath);
    loader.loadAll();
    loader.loadTrainingBatch(batch);

    std::cout << "Test training batch: " << std::endl << "Inputs:" << std::endl;
    std::cout << batch.getInputs().toString() << std::endl;
    std::cout << "Targets:" << std::endl;
    std::cout << batch.getTargets().toString() << std::endl;
}

void testTensorCopyConstructorHelper(Tensor t2)
{
    std::cout << t2.getId() << std::endl;
}

void testTensorCopyConstructor(Device testDevice)
{
    Tensor t(testDevice, {4, 4});
    fill(t, 1.0f);
    testTensorCopyConstructorHelper(t);
}

void testNumTokens() {
    bool pass = true;
    std::string s = "This, is, string,";
    if(numTokens(s) != 3) pass = false;

    if(!pass) std::cout << "testNumTokens failed." << std::endl;
    else std::cout << "testNumTokens passed." << std::endl;
}