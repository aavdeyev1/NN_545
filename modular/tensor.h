#pragma once
#include <iostream>
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
    TensorShape transposedCopy();
    std::string toString();
};

class Tensor
{
private:
	static int nextId;
	int id;
    Device device;
    TensorShape shape;
	
public:
	float *value;
    Tensor(Device device, TensorShape shape);
	std::string toString();
    TensorShape getShape();
    Device getDevice();

    void move(Device newDevice);

	int ndims();
	int getId();
	int dataLength();
	
~Tensor();
};