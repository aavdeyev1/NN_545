#include "tensor.h"

int Tensor::nextId = 0;

TensorShape::TensorShape(int x): x(x), y(0), z(0), w(0) {}
TensorShape::TensorShape(int x, int y): x(x), y(y), z(0), w(0) {}
TensorShape::TensorShape(int x, int y, int z) : x(x), y(y), z(z), w(0) {}
TensorShape::TensorShape(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}

int TensorShape::ndims() {
    if(w != 0) return 4;
    if(w == 0 && z != 0) return 3;
    if(z == 0 && y != 0) return 2;
    if(y == 0 && x != 0) return 1;
    if(x == 0) return 0;
    return 0;
}

bool TensorShape::equals(TensorShape otherShape) {
    return  x == otherShape.x &&
            y == otherShape.y &&
            z == otherShape.z &&
            w == otherShape.w;
}

int TensorShape::matrixHeight() {
    if(ndims() == 1) return 0;
    if(ndims() == 2) return x;
    if(ndims() == 3) return y;
    if(ndims() == 4) return z;
    return -1;
}

int TensorShape::matrixWidth() {
    if(ndims() == 1) return x;
    if(ndims() == 2) return y;
    if(ndims() == 3) return z;
    if(ndims() == 4) return w;
    return -1;
}

TensorShape TensorShape::transposedCopy() {
    // Swap last two dimensions
    if(ndims() == 1) return  TensorShape(x);
    if(ndims() == 2) return TensorShape(y, x, 0, 0);
    if(ndims() == 3) return TensorShape(x, z, y, 0);
    if(ndims() == 4) return TensorShape(x, y, w, z);
}

std::string TensorShape::toString() {
    std::stringstream ss;
    ss << "(" << x;
    if(ndims() > 1) ss << ", " << y;
    if(ndims() > 2) ss << ", " << z;
    if(ndims() > 3) ss << ", " << w;
    ss << ")";
    return ss.str();
}

Tensor::Tensor(Device device, TensorShape shape):
    id(++Tensor::nextId),
    device(device),
	shape(shape),
	value(NULL)
{
	if(device == CPU)
	{
        if(this->dataLength() > 0) {
            value = (float *) malloc(this->dataLength() * sizeof(float));
        }
	}
	else if(device == GPU)
	{
		// 🚧🚧🚧 
	}
}

void Tensor::move(Device newDevice)
{
    if(device == newDevice) return;

    if(newDevice == CPU) {
        // todo support gpu
        device = CPU;
    } else if (newDevice == GPU)
    {
        // todo support gpu
        device = GPU;
    }

}


Tensor::~Tensor() 
{
	if(value != NULL) {
		if(device == CPU){
			free(value);
		}
			
		if(device == GPU){
			// 🚧🚧🚧 
		}
	}
}

int Tensor::dataLength()
{
	int len = shape.x;
    if(shape.y != 0) len *= shape.y;
    if(shape.z != 0) len *= shape.z;
    if(shape.w != 0) len *= shape.w;
	return len;
}

TensorShape Tensor::getShape()
{
	return shape;
}

int Tensor::ndims()
{
	return shape.ndims();
}

int Tensor::getId()
{
	return id;
}

std::string Tensor::toString(){
	std::stringstream ss;
	ss << "Tensor(" << std::endl;
	ss << "   id=(" << id << ")" << std::endl;
	ss << "   shape=(" << getShape().x;
    if(getShape().y != 0) ss << ", " << getShape().y;
    if(getShape().z != 0) ss << ", " << getShape().z;
    if(getShape().w != 0) ss << ", " << getShape().w;
    ss << ")," << std::endl;
	
	ss << "   data=(" << std::endl;

	if(value == NULL){
		ss << "NULL";
	} else {
        for(int i = 0; i < shape.ndims() - 1; i++){
            ss << "[";
        }

        int lastDimSize = shape.matrixWidth();
        int secondLastDimSize = shape.matrixHeight();

        for(int i = 0; i < secondLastDimSize; i++)
        {
            ss << "[";
            for(int j = 0; j < lastDimSize; j++){
                int idx = i * lastDimSize + j;
                ss << value[idx];
                if(j != lastDimSize - 1) ss << ", ";
            }

            ss << "]";
            if(i != secondLastDimSize - 1) ss << std::endl;
        }

        for(int i = 0; i < shape.ndims() - 1; i++){
            ss << "]";
        }
        ss << ")" << std::endl;

	}

	ss << std::endl << "    )" << std::endl;
	ss << ")" << std::endl;
	
	return ss.str();
}

Device Tensor::getDevice()
{
    return device;
}