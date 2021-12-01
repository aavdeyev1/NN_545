#include <iostream>
using namespace std;
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <cstring>
#include <iomanip>
#include <limits>
#include <string.h>

#include "cpu.h"
#include "kernels.h"

// Make static array for these numbers so we can grow the number of hidden layers
#define numIn_ 2
#define numH_ 3
#define numOut_ 1

#define numTrainingPair_ 64

void printArray(float *arr, int rows, int cols, int shouldPrint);

int main()
{
    int indata[8][8] = {
                            { 1,1,1,1, 1,1,1,1},
                            { 1,1,1,1, 1,1,1,1},
                            { 1,1,0,1, 1,1,1,1},
                            { 1,0,0,0, 1,1,1,1},
                            { 1,0,0,0, 0,0,1,1},
                            { 0,0,0,0, 0,0,1,1},
                            { 0,0,0,0, 0,1,1,1},
                            { 0,0,0,1, 1,1,1,1}
                        };
    float fx(float); // init activation fn

    // Need linearized input/output for GPU.
    int i,j,k = 0,input[numIn_*numTrainingPair_] = {0},output[numTrainingPair_] = {0};

    for(i = 0; i < 8; i ++)
        for(j = 0; j< 8; j++)
        {
            input[k] = i;
            input[k + 1] = j;
            output[(int)ceil(k / 2)] = indata[i][j];
            k ++;
        }

    printArray(input, numIn_, numTrainingPair_, 1);
    printArray(output, 1, numTrainingPair_, 1);

    // bpNeuralNetwork<int> myBPNN;
    // myBPNN.training( input,output,64,0.02f,100000l,fx);
    cout << "\n\n\n                Press any key to exit!";
    getchar();
    return 0;
}


// Make cudaMemcpy and cudaMalloc to allocate memory for gpu
// Input will be the input and output arrays calculated in main numTrainSample, learnRate, long maxNumTrainIterate, *pLogisticFun
void runGPU() {

}


void printArray(int *arr, int rows, int cols, int shouldPrint){
    if (!shouldPrint)
       return;
           
    int i,j;
 
    for(i=0; i<rows; i++){
       for(j=0; j<cols; j++){
       
          printf("%04.2f ", arr[i*cols + j]);
       }
       printf("\n");
    }
 
    printf("\n");
 }