#include "layer.h"

Layer::~Layer() {
    if(outputCache != NULL) delete outputCache;
    if(variables != NULL) delete variables;
    if(inputGradsCache != NULL) delete inputGradsCache;

}

Layer::Layer(Device device, Layer* inputLayer, TensorShape outputShape, TensorShape gradsCacheShape):
name("unnamed"),
device(device),
inputLayer(inputLayer),
outputShape(outputShape),
gradsCacheShape(gradsCacheShape),
outputCache(NULL),
variables(NULL),
inputGradsCache(NULL)
 {

 }

 void Layer::setOutputLayer(Layer *newOutputLayer) {
    outputLayer = newOutputLayer;
}

void Layer::forward() {}

void Layer::backward() {}

std::string Layer::getName() { return name; }

void Layer::setName(std::string newName) { name = newName; }

Device Layer::getDevice() {
    return device;
}

void Layer::ensureCachesAllocated() {
    TensorShape outputShape = getOutputShape();
    if(outputCache == NULL){
        std::cout << "Input layer device: " << inputLayer->getDevice() << std::endl;
        std::cout << "Output shape: " << outputShape.toString() << std::endl;
        outputCache = new Tensor(inputLayer->getDevice(), outputShape);
    } else if (!outputCache->getShape().equals(outputShape))
    {
        delete outputCache;
        outputCache = new Tensor(inputLayer->getDevice(), outputShape);
    }

    if(inputGradsCache == NULL && !gradsCacheShape.equals(TensorShape(0))){
        inputGradsCache = new Tensor(inputLayer->getDevice(), gradsCacheShape);
    } else if(!gradsCacheShape.equals(inputGradsCache->getShape())){
        delete inputGradsCache;
        inputGradsCache = new Tensor(inputLayer->getDevice(), gradsCacheShape);
    }

}

TensorShape Layer::getOutputShape()
{
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
    ss << "    Variable data: ";
    if(variables == NULL) ss << "NULL" << std::endl;
    else ss << std::endl << variables->toString() << std::endl;
    ss << "    Output cache allocated: ";
    if(outputCache != NULL) ss << "true" << std::endl;
    else ss << "false" << std::endl;
    ss << "    Gradient data cache allocated: ";
    if(inputGradsCache != NULL) ss << "true" << std::endl;
    else ss << "false" << std::endl;

    return ss.str();
}

/// Input layer

InputLayer::InputLayer(Device device, TensorShape inputShape) :
    Layer(device, NULL, inputShape, TensorShape(0)), inputShape(inputShape)
{
}

void InputLayer::setData(Tensor inputData) {
    if(!inputData.getShape().equals(inputShape)){
        std::cout << "ERROR: Wrong input data shape for InputLayer." << std::endl;
        exit(1);
    }
    outputCache = &inputData;
}

TensorShape InputLayer::getOutputShape() {
    return inputShape;
}

void InputLayer::forward(){
    outputLayer->forward();
}

void InputLayer::backward() { }

/// LinearLayer

LinearLayer::LinearLayer(Layer *inputLayer, int outputDim) :
Layer(inputLayer->getDevice(), inputLayer, TensorShape(0), TensorShape(0)),
outputDim(outputDim)
{
    inputLayer->setOutputLayer(this);
    outputShape = calcOutputShape(inputLayer->getOutputShape(), outputDim);
    gradsCacheShape = outputShape;
    initWeights();
}

TensorShape LinearLayer::calcOutputShape(TensorShape inputShape, int outputDim)
{
    int batchSize;
    if(inputLayer->getOutputShape().ndims() == 2) {
    // Batched data
        batchSize = inputLayer->getOutputShape().x;
    } else {
    // todo: handle input not 2D matrix
    }
    return TensorShape(batchSize, outputDim);
}

TensorShape LinearLayer::calcWeightsShape() {
    int inputFeatures = inputLayer->getOutputShape().y;
    return TensorShape(inputFeatures, outputDim);
}

void LinearLayer::initWeights() {
    // Set weights using Glorot Uniform initialization

    if(variables == NULL) {
        variables = new Tensor(getDevice(), calcWeightsShape());
    }

    int featuresIn = variables->getShape().matrixHeight();
    int featuresOut = variables->getShape().matrixWidth();

    float sd = (float) sqrt( 6.0 / (featuresIn + featuresOut));

    // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-sd, sd);

    for(int i = 0; i < variables->dataLength(); i++)
    {
        variables->value[i] = dis(gen);
    }
}

void LinearLayer::forward() {
    ensureCachesAllocated();
    matmul(inputLayer->outputCache, variables, outputCache);
    outputLayer->forward();
}

void LinearLayer::backward() {
    ensureCachesAllocated();
    // Gradient of weights w.r.t. input value X = transposed(X)
     transposedCopy(inputLayer->outputCache, inputGradsCache);
     inputLayer->backward();
}

TensorShape LinearLayer::getOutputShape() {
    outputShape = calcOutputShape(inputLayer->getOutputShape(), outputDim);
    return outputShape;
}

// Sigmoid layer
SigmoidLayer::SigmoidLayer(Layer *inputLayer) :
Layer(inputLayer->getDevice(), inputLayer, inputLayer->getOutputShape(), inputLayer->getOutputShape())
{
    inputLayer->setOutputLayer(this);
}

void SigmoidLayer::forward() {
    ensureCachesAllocated();
    sigmoidFunction(inputLayer->outputCache, outputCache, true);
    outputLayer->forward();
}

void SigmoidLayer::backward() {
    ensureCachesAllocated();
    sigmoidFunction(inputLayer->outputCache, inputGradsCache, false);
    inputLayer->backward();
}

TensorShape SigmoidLayer::getOutputShape() {
    outputShape = inputLayer->getOutputShape();
    return outputShape;
}

LossLayer::LossLayer(Layer *inputLayer):
    Layer(inputLayer->getDevice(), inputLayer, TensorShape(0), inputLayer->getOutputShape()),
    targets(NULL)
{
    inputLayer->setOutputLayer(this);
    setOutputLayer(NULL);
}

LossLayer::~LossLayer() {
    if(targets != NULL) delete targets;
}

void LossLayer::setTargets(Tensor *newTargets) {
    if(targets != NULL) delete targets;
    targets = newTargets;
}

ModelMetrics LossLayer::getMetrics()
{
    return metrics;
}

void LossLayer::forward() {
    if(!targets->getShape().equals(gradsCacheShape) && targets->getShape().equals(inputLayer->getOutputShape()))
    {
        if(inputGradsCache != NULL) delete inputGradsCache;
        inputGradsCache = new Tensor(getDevice(), targets->getShape());
    } else if (!targets->getShape().equals(gradsCacheShape) && !targets->getShape().equals(inputLayer->getOutputShape())) {
        std::cout << "ERROR: Incomptaible shapes from input tensor & target tensor." << std::endl;
        exit(1);
    }
    metrics = ModelMetrics{mse(targets, inputLayer->outputCache)};
}

void LossLayer::backward() {
    ensureCachesAllocated();
    MSEGradient(targets, inputLayer->outputCache, inputGradsCache);
    inputLayer->backward();
}

TensorShape LossLayer::getOutputShape(){
    return NULL;
}