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
    int trainIters = 1;
    float lr = 0.01f;
    int batchSize = 256;
    int nInputFeatures = 8;
    int nOutputFeatures = 1;

    TensorShape batchInputsShape(batchSize, nInputFeatures);
    TensorShape batchTargetsShape(batchSize, nOutputFeatures);
    TrainBatch batch(batchInputsShape, batchTargetsShape);

    std::string xTrainPath = "datasets/california_housing_prepared/x_train.csv";
    std::string xTestPath = "datasets/california_housing_prepared/x_test.csv";
    std::string yTrainPath = "datasets/california_housing_prepared/y_train.csv";
    std::string yTestPath = "datasets/california_housing_prepared/y_test.csv";

    CSVTrainTestDataLoader loader(xTrainPath, xTestPath, yTrainPath, yTestPath);
    loader.loadAll();

    InputLayer layer1("Input", device, batchInputsShape);
    LinearLayer layer2("Linear1", &layer1, 8);
    SigmoidLayer layer3("Sigmoid1", &layer2);
    LinearLayer layer4("Linear2", &layer3, 8);
    SigmoidLayer layer5("Sigmoid2", &layer4);
    LinearLayer layer6("Linear3", &layer5, 1);
    SigmoidLayer layer7("Sigmoid3", &layer6);
    LossLayer outputLayer("MSELoss", &layer7);

    std::cout << "Model:" << std::endl;
    std::cout << layer1.toString() << std::endl;
    std::cout << layer2.toString() << std::endl;
    std::cout << layer3.toString() << std::endl;
    std::cout << layer4.toString() << std::endl;
    std::cout << layer5.toString() << std::endl;
    std::cout << layer6.toString() << std::endl;
    std::cout << layer7.toString() << std::endl;
    std::cout << outputLayer.toString() << std::endl;


    loader.loadTrainingBatch(batch);
    layer1.setData(batch.getX());
    outputLayer.setTargets(batch.getY());

    layer1.forward();
    MSELoss = outputLayer.output.item();
    std::cout << "Loss: " << MSELoss << std::endl;
    outputLayer.backward();
}