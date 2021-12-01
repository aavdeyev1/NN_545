#pragma  once
#include <string>
#include <math.h>
#include <memory>
#include <random>
#include "tensor.h"
#include "tensor_ops.h"

struct Layer
{
private:
    std::string name;
    Device device;
protected:
    Layer(std::string name, Device device, Layer* inputLayer, TensorShape inputShape, TensorShape outputShape);
    TensorShape inputShape;
    TensorShape outputShape;

public:
    Layer * inputLayer;
    Layer * outputLayer;

    Tensor output;


    virtual void forward();
    virtual void backward();

    std::string getName();
    std::string toString();
    //void setName(std::string newName);
    void setOutputLayer(Layer* outputLayer);
    TensorShape getOutputShape();
    Device getDevice();
};

struct InputLayer : public Layer
{
public:
    InputLayer(std::string name, Device device, TensorShape inputShape);
    void setData(Tensor inputData);

    virtual void forward();
    virtual void backward();
};

struct LinearLayer : public Layer
{
private:
    void initWeights();
    Tensor weights;
    Tensor bias;

public:
    LinearLayer(std::string name, Layer * inputLayer, int outputDim);
    virtual void forward();
    virtual void backward();
};

struct SigmoidLayer : public Layer
{
public:
    SigmoidLayer(std::string name, Layer* inputLayer);
    virtual void forward();
    virtual void backward();
};

struct LossLayer : public Layer
{
private:
    Tensor targets;
public:
    LossLayer(std::string name, Layer* inputLayer);
    void setTargets(Tensor targets);
    virtual void forward();
    virtual void backward();
};