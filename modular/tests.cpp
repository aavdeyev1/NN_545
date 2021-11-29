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
    testGetTrainBatch(testDevice);
}

void testAddTensors(Device testDevice)
{
    if(testDevice == CPU) {
        TensorShape shape = {4};
        Tensor *A = new Tensor(CPU, shape);
        fill(A, 1.0f);


        Tensor *B = new Tensor(CPU, shape);
        fill(B, 2.0f);

        Tensor *C = new Tensor(CPU, shape);

        addTensors(A, B, C);

        for (int i = 0; i < C->dataLength(); i++) {
            if (C->value[i] != 3.0f) {
                std::cout << "testAddTensors failed." << std::endl;
                exit(1);
            }
        }

        delete A;
        delete B;
        delete C;
    } else if(testDevice == GPU){
        // todo support gpu
    }

    std::cout << "testAddTensors passed." << std::endl;
}

void testTensorSmoke(Device testDevice)
{
    if(testDevice == CPU){
        TensorShape shape = {3, 4};
        Tensor* A = new Tensor(CPU, shape);
        delete A;
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

        InputLayer *inputLayer = new InputLayer(CPU, inputLayerShape);

        LinearLayer *linearLayer = new LinearLayer(inputLayer, 8);

        if (!linearLayer->getOutputShape().equals(linearLayerShape)) {
            std::cout << "testLayerSmoke failed shape test." << std::endl;
            exit(1);
        }

        delete inputLayer;
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
        Tensor* A = new Tensor(CPU, shapeA);
        fill(A, 1.0f);


        Tensor* B = new Tensor(CPU, shapeB);
        fill(B, 2.0f);
        Tensor * C = new Tensor(CPU, shapeC);

        matmul(A, B, C);
        delete A;
        delete B;
        delete C;
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
        Tensor* A = new Tensor(CPU, shape);
        A->value[0] = 1;
        A->value[1] = 2;
        A->value[2] = 3;
        A->value[3] = 4;

        Tensor* B = new Tensor(CPU, shape);
        B->value[0] = -4;
        B->value[1] = -3;
        B->value[2] = -2;
        B->value[3] = -1;

        Tensor* C = new Tensor(CPU, shape);

        matmul(A, B, C);
        if( C->value[0] != -8 ||
            C->value[1] != -5 ||
            C->value[2] != -20 ||
            C->value[3] != -13){
            std::cout << "testMatmulCorrect failed first test. Got value:" <<std::endl;
            std::cout << C->toString() << std::endl;
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
        Tensor* A = new Tensor(CPU, shapeA);
        Tensor* B = new Tensor(CPU, shapeB);
        A->value[0] = 1;
        A->value[1] = 2;
        A->value[2] = 3;
        A->value[3] = -1;
        A->value[4] = -2;
        A->value[5] = -3;
        transposedCopy(A, B);
        if( B->value[0] != 1 &&
            B->value[1] != -1 &&
            B->value[2] != 2 &&
            B->value[3] != -2 &&
            B->value[4] != 3 &&
            B->value[5] != -3
                ){
            std::cout << "testTransposedCopyCorrect failed, value of out matrix:" << std::endl;
            std::cout << B->toString() << std::endl;
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
    std::vector<std::vector<std::string>> cells = loadCSVCells(path);
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
}

void testGetTrainBatch(Device testDevice)
{
    TrainBatch batch;
    int batchSize = 12;
    std::string xTrainPath = "datasets/california_housing_prepared/x_train.csv";
    std::string xTestPath = "datasets/california_housing_prepared/x_test.csv";
    std::string yTrainPath = "datasets/california_housing_prepared/y_train.csv";
    std::string yTestPath = "datasets/california_housing_prepared/y_test.csv";

    CSVTrainTestDataLoader loader(xTrainPath, xTestPath, yTrainPath, yTestPath);
    loader.loadAll();
    loader.loadTrainingBatch(&batch, testDevice, batchSize);

    std::cout << "Test training batch: " << std::endl << "Inputs:" << std::endl;
    std::cout << batch.inputs->toString() << std::endl;
    std::cout << "Targets:" << std::endl;
    std::cout << batch.targets->toString() << std::endl;

    delete batch.inputs;
    delete batch.targets;
}