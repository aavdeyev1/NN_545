#include "tensor_ops.h"

void transposedCopy(Tensor* in, Tensor* out) {
    // todo support 4D matrices
    Device device = in->getDevice();
    if (out->getDevice() != device) {
        std::cout << "Error: In & out tensors must be on same device for transposedCopy." << std::endl;
        exit(1);
    }
    if (device == CPU) {
        transposedCopyMatrix2DCPU(in->value, out->value, in->getShape().matrixHeight(), in->getShape().matrixWidth());
    } else if(device == GPU){
        // todo support GPU
    }
}

void fill(Tensor* t, float val){
    int n = t->dataLength();

    if(t->getDevice() == CPU){
        for(int i = 0; i < n; i++)
            t->value[i] = val;
    }

    if(t->getDevice() == GPU){
        // todo support gpu
    }
}

void addTensors(Tensor* A, Tensor* B, Tensor *out)
{
    TensorShape shape = A->getShape();
    if(!B->getShape().equals(shape))
    {
        std::cout << "FAILURE: Shape of tensor A doesn't match tensor B.";
        exit(1);
    }
    if(!out->getShape().equals(shape))
    {
        std::cout << "FAILURE: Shape of tensor A doesn't match tensor C.";
        exit(1);
    }
    unsigned int n = A->dataLength();
    for(unsigned int i = 0; i < n; i++)
    {
        out->value[i] = A->value[i] + B->value[i];
    }
}

void matmul(Tensor* A, Tensor* B, Tensor* out)
{
    Device device = A->getDevice();

    if(B->getDevice() != device || out->getDevice() != device)
    {
        std::cout << "Error: Can't multiply matrices on different devices." << std::endl;
        exit(1);
    }
    TensorShape AShape = A->getShape();
    TensorShape BShape = B->getShape();
    if(AShape.ndims() != BShape.ndims()) {
        std::cout << "ERROR: in matmul, A and B must match number of dimensions." << std::endl;
        exit(1);
    }

    if(AShape.matrixWidth() != BShape.matrixHeight()) {
        std::cout << "ERROR: In MxN @ NxP matrix multiplication, dimensions M and P must match." << std::endl;
        exit(1);
    }
    TensorShape outShape = AShape;
    if(outShape.w != 0) {
        // 4D data
        if(AShape.x != BShape.x || AShape.y != BShape.y) {
            std::cout << "ERROR: in matmul, dimensions 0 and 1 must match for A and B." << std::endl;
            exit(1);
        }
        outShape.w = BShape.matrixWidth();
        if(device == CPU) {
            // todo implement 4D matrix multiplication
        }
        if(device == GPU) {
            // todo support gpu
        }
    }
    else if(outShape.z != 0) {
        if(AShape.x != BShape.x) {
            std::cout << "ERROR: in matmul, dimension 0 must match for A and B." <<std::endl;
            exit(1);
        }
        if(device == CPU){
            // todo implement 3D matrix multiplication
        }
        if(device == GPU) {
            //todo support gpu
        }
    }
    else if(outShape.y != 0) {
        // 2D data
        outShape.y = BShape.matrixWidth();
        // TODO support GPU
        if(device == CPU){
            matmul2DCPU(A->value, B->value, out->value, AShape.x, AShape.y, BShape.y);
        }
        if(device == GPU) {
            //todo support GPU
        }
    }
}

void sigmoidFunction(Tensor* in, Tensor* out, bool forward)
{
    Device device = in->getDevice();
    if(device != out->getDevice()){
        std::cout << "ERROR: Devices of input and output tensors must match in sigmoidFunction" <<std::endl;
        exit(1);
    }
    if(!in->getShape().equals(out->getShape())){
        std::cout << "ERROR: In & out tensors must have same shape in sigmoid function." << std::endl;
        exit(1);
    }
    if(device == CPU){
        sigmoidFunctionElementwiseCPU(in->value, out->value, in->dataLength(), forward);
    } else if (device == GPU){
        //todo support GPU
    }
}

void MSEGradient(Tensor* targets, Tensor* predictions, Tensor* out)
{
    Device device = targets->getDevice();
    if(device != predictions->getDevice() || device != out->getDevice()){
        std::cout << "ERROR: All tensors must be on same device." << std::endl;
        exit(1);
    }

    TensorShape shape = predictions->getShape();
    if(!targets->getShape().equals(shape) || !out->getShape().equals(shape)){
        std::cout << "ERROR: All tensors must have the same shape." << std:: endl;
        exit(1);
    }

    if(device == CPU){
        for(int i = 0; i < out->dataLength(); i++){
            out->value[i] = -2.0f * (targets->value[i] - predictions->value[i]);
        }
    }
    if(device == GPU){
        // todo support GPU
    }
}