#include "tensor_ops.h"

void transposedCopy(Tensor in, Tensor out) {
    // todo support 4D matrices
    Device device = in.getDevice();
    if (out.getDevice() != device) {
        cout << "Error: In & out tensors must be on same device for transposedCopy." << endl;
        exit(1);
    }
    if (device == CPU) {
        transposedCopyMatrix2DCPU(in.data(), out.data(), in.getShape().matrixHeight(), in.getShape().matrixWidth());
    } else if(device == GPU){
        // todo support GPU
    }
}

void fill(Tensor t, float val){
    int n = t.dataLength();

    if(t.getDevice() == CPU){
        for(int i = 0; i < n; i++)
            t.data()[i] = val;
    }

    if(t.getDevice() == GPU){
        // todo support gpu
    }
}

void addTensors(Tensor A, Tensor B, Tensor out)
{
    TensorShape shape = A.getShape();
    if(!B.getShape().equals(shape))
    {
        cout << "FAILURE: Shape of tensor A doesn't match tensor B." << endl;
        exit(1);
    }
    if(!out.getShape().equals(shape))
    {
        cout << "FAILURE: Shape of tensor A doesn't match tensor C." << endl;
        exit(1);
    }
    int n = A.dataLength();
    for(unsigned int i = 0; i < n; i++)
    {
        out.data()[i] = A.data()[i] + B.data()[i];
    }
}

void matmul(Tensor A, Tensor B, Tensor out)
{
    Device device = A.getDevice();

    if(B.getDevice() != device || out.getDevice() != device)
    {
        cout << "Error: Can't multiply matrices on different devices." << endl;
        exit(1);
    }
    TensorShape AShape = A.getShape();
    TensorShape BShape = B.getShape();

    if(AShape.ndims() != BShape.ndims()) {
        cout << "ERROR: in matmul, A and B must match number of dimensions." << endl;
        exit(1);
    }

    int m = AShape.x;
    int n = AShape.y;
    int p = BShape.y;
    if(AShape.y != n) {
        std::cout << "ERROR: Matrices of dimensions MxN and NxP must match on inner dimension." << std::endl;
        exit(1);
    }

    if(device == CPU){
        matmul2DCPU(A.data(), B.data(), out.data(), m, n, p);
    }
    if(device == GPU) {
        //todo support GPU
    }
}

void sigmoidFunction(Tensor in, Tensor out, bool forward)
{
    Device device = in.getDevice();
    if(device != out.getDevice()){
        std::cout << "ERROR: Devices of input and output tensors must match in sigmoidFunction" <<std::endl;
        exit(1);
    }
    if(!in.getShape().equals(out.getShape())){
        std::cout << "ERROR: In & out tensors must have same shape in sigmoid function." << std::endl;
        exit(1);
    }
    if(device == CPU){
        sigmoidFunctionElementwiseCPU(in.data(), out.data(), in.dataLength(), forward);
    } else if (device == GPU){
        //todo support GPU
    }
}

void MSEGradient(Tensor targets, Tensor predictions, Tensor out)
{
    Device device = targets.getDevice();
    if(device != predictions.getDevice() || device != out.getDevice()){
        cout << "ERROR: All tensors must be on same device." << endl;
        exit(1);
    }

    TensorShape shape = predictions.getShape();
    if(!targets.getShape().equals(shape) || !out.getShape().equals(shape)){
        cout << "ERROR: All tensors must have the same shape." << endl;
        exit(1);
    }

    if(device == CPU){
        for(int i = 0; i < out.dataLength(); i++){
            out.data()[i] = -2.0f * (targets.data()[i] - predictions.data()[i]);
        }
    }
    if(device == GPU){
        // todo support GPU
    }
}

void MSE(Tensor targets, Tensor predictions, Tensor out)
{
    float sum = 0.0f;
    Device device = targets.getDevice();
    if(device != predictions.getDevice() || device != out.getDevice()){
        cout << "ERROR: All tensors must be on same device." << endl;
        exit(1);
    }

    TensorShape shape = predictions.getShape();
    if(!targets.getShape().equals(shape)){
        cout << "ERROR: Targets and predictions must have the same shape." << endl;
        exit(1);
    }

    if(!out.getShape().equals(TensorShape(1))){
        cout << "ERROR: Output tensor must be an item." << endl;
        exit(1);
    }

    if(device == CPU){
        for(int i = 0; i < out.dataLength(); i++){
            sum += sqrt(pow(targets.data()[i] - predictions.data()[i], 2.0));
        }
        sum /= (float) out.dataLength();
        out.data()[0] = sum;
    }
    if(device == GPU){
        // todo support GPU
    }
}