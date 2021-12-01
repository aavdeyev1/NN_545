#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include "device.h"

struct TensorShape
{
    TensorShape(int x);
    TensorShape(int x, int y);
    TensorShape(int x, int y, int z);
    TensorShape(int x, int y, int z, int w);
    int x, y, z, w;
    int ndims();
    bool equals(TensorShape otherShape);
    int matrixWidth();
    int matrixHeight();
    TensorShape transposed();
    std::string toString();
};

class Tensor
{
private:
	static int nextId;
	int id;
    int* referenceCount;
    Device device;
    TensorShape shape;
    bool isShallowCopy;
    float* value;
	
public:
    ~Tensor();
    Tensor(Device device, TensorShape shape);
    Tensor(const Tensor &copyObj);
    void resizeData(TensorShape shape);

    std::string toString();
    TensorShape getShape();
    Device getDevice();
    float* data();

    void move(Device newDevice);

	int ndims();
	int getId();
	int dataLength();

    float item();
};