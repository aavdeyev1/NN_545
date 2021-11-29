#include "metrics.h"

float mse(Tensor * real, Tensor * predicted)
{
    float mseSum = 0;
    int n;

    Device  device = real->getDevice();
    TensorShape shape = real->getShape();
    if(!predicted->getShape().equals(shape)) {
        std::cout << "Two input tensors must be the same shape to calculate RMSE." << std::endl;
        exit(1);
    }
    if(predicted->getDevice() != device) {
        std::cout << "Two input tensors must be on same device to calculate RMSE." << std::endl;
        exit(1);
    }

    if(device == CPU){
        n = real->dataLength();
        for(int i = 0; i < n; i++){
            mseSum += pow(real->value[i] - predicted->value[i], 2.0f);
        }
        return mseSum / n;
    }
    if(device == GPU){
        // todo support GPU
    }
    return 0.0;
}

float rmse(Tensor * real, Tensor * predicted)
{
    return sqrt(mse(real, predicted));
}