//This Kernel will be dedicated to forward feeding and backward propigation which gets error and new weights for one training pair.
// The input will be...	void training(T trainData[64][numIn_],int trueOut[64][numOut_],const int numTrainSample,
                      //const float learnRate,const long maxNumTrainIterate,float (*pLogisticFun)(float))

//The output will be the updated weights for each training pair



//This Kernel will be dedicated to updating the weights and biases for one batch

#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_DIM 4                     // Tile dimension
#define DIMX 1                            
#define DIMY 2
#define DIMZ 3

void printArray(float *arr, int rows, int cols, int shouldPrint);

__global__ void MatMulNoShared(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n) 
            if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

int main() {

    int CCols = DIMZ, CRows=DIMX, ACols=DIMY, ARows=DIMX, BCols=DIMZ, BRows=DIMY;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (CRows + dimBlock.y - 1)/dimBlock.y;

    float *deviceA, *deviceB, *deviceC;

    float* hostA    = (float*)malloc(DIMX*DIMY*sizeof(float));
    float* hostB    = (float*)malloc(DIMY*DIMZ*sizeof(float));
    float* hostC    = (float*)malloc(DIMX*DIMZ*sizeof(float));
    float* hostCp   = (float*)malloc(DIMX*DIMZ*sizeof(float));

    // for (int x = 0; x<DIMX; x++)
    //     for (int y = 0; y<DIMY; y++) {
    //         hostA[x*DIMY+y] = rand()/(float)RAND_MAX;
    //         hostB[x*DIMY+y] = rand()/(float)RAND_MAX;
    //     }

    hostA[0] = 1.0;
    hostA[1] = 2.0;
    hostB[0] = 1.0;
    hostB[1] = 2.0;
    hostB[3] = 3.0;
    hostB[4] = 1.0;
    hostB[5] = 2.0;
    hostB[6] = 3.0;

    cudaMalloc((void **)&deviceA, DIMX*DIMY*sizeof(float));
    cudaMalloc((void **)&deviceB, DIMY*DIMZ*sizeof(float));
    cudaMalloc((void **)&deviceC, DIMX*DIMZ*sizeof(float));

    cudaMemcpy(deviceA, hostA, DIMX*DIMY*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, DIMY*DIMZ*sizeof(float), cudaMemcpyHostToDevice);

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , deviceC , ARows , ACols, BRows ,BCols , CRows , CCols);

    cudaMemcpy(hostC, deviceC, DIMX*DIMZ*sizeof(float), cudaMemcpyDeviceToHost);
    printArray(hostA, DIMX, DIMY, 1);
    printArray(hostB, DIMY, DIMZ, 1);
    printArray(hostC, DIMX, DIMZ, 1);

    return 0;
}


void printArray(float *arr, int rows, int cols, int shouldPrint){
    if (!shouldPrint)
       return;
           
    int i,j;
 
    for(i=0; i<rows; i++){
       for(j=0; j<cols; j++){
       
          printf("%d ", arr[i*cols + j]);
       }
       printf("\n");
    }
 
    printf("\n");
 }