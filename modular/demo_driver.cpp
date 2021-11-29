#include "demo_driver.h"

int main()
{
    runAllTests(CPU);
    //runAllTests(GPU); todo
    demoCaliforniaHousingDataset(CPU);
    return 0;
}

void demoCaliforniaHousingDataset(Device device)
{
    float MSELoss;
    TrainBatch batch;

    int trainIters = 1;
    float lr = 0.01f;
    int batchSize = 256;
    int nInputFeatures = 8;

    TensorShape batchShape(batchSize, nInputFeatures);

    std::string xTrainPath = "datasets/california_housing_prepared/x_train.csv";
    std::string xTestPath = "datasets/california_housing_prepared/x_test.csv";
    std::string yTrainPath = "datasets/california_housing_prepared/y_train.csv";
    std::string yTestPath = "datasets/california_housing_prepared/y_test.csv";

    CSVTrainTestDataLoader loader(xTrainPath, xTestPath, yTrainPath, yTestPath);
    loader.loadAll();

    InputLayer layer1(device, batchShape);
    layer1.setName("Input");

    LinearLayer layer2(&layer1, 8);
    layer2.setName("Linear1");
    SigmoidLayer layer3(&layer2);
    layer3.setName("Sigmoid1");

    LinearLayer layer4(&layer3, 8);
    layer4.setName("Linear2");
    SigmoidLayer layer5(&layer4);
    layer5.setName("Sigmoid2");

    LinearLayer layer6(&layer5, 1);
    layer6.setName("Linear3");
    SigmoidLayer layer7(&layer6);
    layer7.setName("Sigmoid3");

    LossLayer outputLayer(&layer7);
    outputLayer.setName("MSELoss");

    std::cout << "Model:" << std::endl;
    std::cout << layer1.toString() << std::endl;
    std::cout << layer2.toString() << std::endl;
    std::cout << layer3.toString() << std::endl;
    std::cout << layer4.toString() << std::endl;
    std::cout << layer5.toString() << std::endl;
    std::cout << layer6.toString() << std::endl;
    std::cout << layer7.toString() << std::endl;
    std::cout << outputLayer.toString() << std::endl;

    loader.loadTrainingBatch(&batch, device, batchSize);
    layer1.setData(*batch.inputs);
    outputLayer.setTargets(batch.targets);

    layer1.forward();
    MSELoss = outputLayer.getMetrics().mse;
    std::cout << "Loss: " << MSELoss << std::endl;
    outputLayer.backward();
}