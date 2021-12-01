#include "layer.h"

Layer::Layer(std::string name,
             Device device,
             Layer* inputLayer,
             TensorShape inputShape,
             TensorShape outputShape):
name(name),
device(device),
inputLayer(inputLayer),
inputShape(inputShape),
outputShape(outputShape),
output(device, outputShape)
 { }

 void Layer::setOutputLayer(Layer *newOutputLayer) {
    outputLayer = newOutputLayer;
}

void Layer::forward() {}

void Layer::backward() {}

std::string Layer::getName() { return name; }

Device Layer::getDevice() {
    return device;
}

TensorShape Layer::getOutputShape(){
    return outputShape;
}

std::string Layer::toString() {
    std::stringstream ss;

    ss << "Layer: " << std::endl;
    ss << "    Name: " << getName() << std::endl;
    ss << "    Device: ";
    if(device == CPU) ss << "CPU" << std::endl;
    else if(device == GPU) ss << "GPU" << std::endl;
    if(inputLayer == NULL) ss << "    Input layer: NULL" << std::endl;
    else ss << "    Input layer: " << inputLayer->getName() << std::endl;

    if(outputLayer == NULL) ss << "    Output layer: NULL" << std::endl;
    else ss << "    Output layer: " << outputLayer->getName() << std::endl;

    ss << "    Output shape: " << outputShape.toString() << std::endl;

    return ss.str();
}

/// Input layer

InputLayer::InputLayer(std::string name, Device device, TensorShape inputShape) :
    Layer(name, device, NULL, inputShape, inputShape) { }

void InputLayer::setData(Tensor inputData) {
    if(!inputData.getShape().equals(inputShape)){
        std::cout << "ERROR: Wrong input data shape for InputLayer." << std::endl;
        exit(1);
    }
    if(inputData.getDevice() == getDevice()){
        output = inputData;
    } else {
        //todo support moving data onto the GPU here
    }

}

void InputLayer::forward(){
    if(outputLayer != NULL) {
        outputLayer->forward();
    }
}

void InputLayer::backward() { }

/// LinearLayer

LinearLayer::LinearLayer(std::string name, Layer * inputLayer, int outputDim) :
    Layer(name, inputLayer->getDevice(),
          inputLayer,
          inputLayer->getOutputShape(),
          {inputLayer->getOutputShape().x, outputDim}
          ),
          weights(inputLayer->getDevice(), {outputShape.y, outputDim}),
          bias(inputLayer->getDevice(), TensorShape(1))
{
    inputLayer->setOutputLayer(this);
    initWeights();
    fill(bias, 0);
}

void LinearLayer::initWeights() {
    // Set weights using Glorot Uniform initialization
    
    int featuresIn = weights.getShape().matrixHeight();
    int featuresOut =  weights.getShape().matrixWidth();

    float sd = (float) sqrt( 6.0 / (featuresIn + featuresOut));

    // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-sd, sd);

    for(int i = 0; i < weights.dataLength(); i++)
    {
        weights.data()[i] = dis(gen);
    }
}

void LinearLayer::forward() {
    matmul(inputLayer->output, weights, output);
    outputLayer->forward();
}

void LinearLayer::backward() {
    //todo
     inputLayer->backward();
}

// Sigmoid layer
SigmoidLayer::SigmoidLayer(std::string name, Layer *inputLayer) :
Layer(name,
      inputLayer->getDevice(),
      inputLayer,
      inputLayer->getOutputShape(),
      inputLayer->getOutputShape())
{
    inputLayer->setOutputLayer(this);
}

void SigmoidLayer::forward() {
    sigmoidFunction(inputLayer->output, output, true);
    outputLayer->forward();
}

void SigmoidLayer::backward() {
    // sigmoidFunction(inputLayer->output, tensor of grads, false);
    inputLayer->backward();
}

LossLayer::LossLayer(std::string name, Layer *inputLayer):
    Layer(name,
          inputLayer->getDevice(),
          inputLayer,
          inputLayer->getOutputShape(),
          TensorShape(1)),
    targets(inputLayer->getDevice(), inputLayer->getOutputShape())
{
    inputLayer->setOutputLayer(this);
    setOutputLayer(NULL);
}

void LossLayer::setTargets(const Tensor newTargets) {
    targets = newTargets;
}

void LossLayer::forward() {
    MSE(targets, inputLayer->output, output);
    MSE(targets, inputLayer->output, output);
}

void LossLayer::backward() {
    // todo
    inputLayer->backward();
}