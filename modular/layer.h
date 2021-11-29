#pragma  once
#include <string>
#include <math.h>
#include <random>
#include "metrics.h"
#include "tensor.h"
#include "tensor_ops.h"

struct Layer
{
private:
    std::string name;
    Device device;
protected:
    Layer(Device device, Layer* inputLayer, TensorShape outputShape, TensorShape gradsCacheShape);
    void ensureCachesAllocated();
    Layer * outputLayer;

    TensorShape outputShape;
    TensorShape gradsCacheShape;

public:
    ~Layer();
    Layer* inputLayer;
    Tensor* outputCache;
    Tensor* variables;
    Tensor* inputGradsCache;

    virtual void forward();
    virtual void backward();

    std::string getName();
    std::string toString();
    void setName(std::string newName);
    void setOutputLayer(Layer* outputLayer);
    virtual TensorShape getOutputShape();
    Device getDevice();
};

struct InputLayer : public Layer
{
public:
    TensorShape inputShape;
    InputLayer(Device device, TensorShape inputShape);
    void setData(Tensor inputData);

    virtual void forward();
    virtual void backward();
    virtual TensorShape getOutputShape();
};

struct LinearLayer : public Layer
{
private:
    void initWeights();
    int outputDim;
    TensorShape calcOutputShape(TensorShape inputShape, int outputDim);
    TensorShape calcWeightsShape();

public:
    LinearLayer(Layer* inputLayer, int outputDim);

    virtual void forward();
    virtual void backward();
    virtual TensorShape getOutputShape();
};

struct SigmoidLayer : public Layer
{
public:
    SigmoidLayer(Layer* inputLayer);
    virtual void forward();
    virtual void backward();
    virtual TensorShape getOutputShape();
};

struct LossLayer : public Layer
{
private:
    Tensor* targets;
    ModelMetrics metrics;
public:
    ~LossLayer();
    LossLayer(Layer* inputLayer);
    void setTargets(Tensor * targets);
    virtual void forward();
    virtual void backward();
    virtual TensorShape getOutputShape();
    ModelMetrics getMetrics();
};