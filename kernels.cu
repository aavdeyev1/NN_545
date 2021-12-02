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

// a (m * n)
// b (n * p)
// c (m * p)

__global__ void MatMulNoShared(float* a, float* b, float* c, int ARows, int Acols, int Acols, int Bcols, int cRows, int ccols) {

    float cValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int col = blockIdx.x*TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + Acols - 1)/TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n) 
            if ((k*TILE_DIM + n < Acols && Row < ARows) && (k*TILE_DIM + n < Acols && col < Bcols))
                cValue += a[Row*Acols + k*TILE_DIM + n] * b[(k*TILE_DIM + n)*Bcols + col];

    }

    if (Row < cRows && col < ccols) c[((blockIdx.y * blockDim.y + threadIdx.y)*ccols)+(blockIdx.x*blockDim.x)+threadIdx.x]=cValue;
}

int main() {

    int ccols = DIMZ, cRows=DIMX, Acols=DIMY, ARows=DIMX, Bcols=DIMZ, Acols=DIMY;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (ccols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (cRows + dimBlock.y - 1)/dimBlock.y;

    float *deviceA, *deviceB, *devicec;

    float* hostA    = (float*)malloc(DIMX*DIMY*sizeof(float));
    float* hostB    = (float*)malloc(DIMY*DIMZ*sizeof(float));
    float* hostc    = (float*)malloc(DIMX*DIMZ*sizeof(float));
    float* hostcp   = (float*)malloc(DIMX*DIMZ*sizeof(float));

    // for (int x = 0; x<DIMX; x++)
    //     for (int y = 0; y<DIMY; y++) {
    //         hostA[x*DIMY+y] = rand()/(float)RAND_MAX;
    //         hostB[x*DIMY+y] = rand()/(float)RAND_MAX;
    //     }

    hostA[0] = 1.0;
    hostA[1] = 2.0;
    hostB[0] = 1.0;
    hostB[1] = 2.0;
    hostB[2] = 3.0;
    hostB[3] = 1.0;
    hostB[4] = 2.0;
    hostB[5] = 3.0;

    cudaMalloc((void **)&deviceA, DIMX*DIMY*sizeof(float));
    cudaMalloc((void **)&deviceB, DIMY*DIMZ*sizeof(float));
    cudaMalloc((void **)&devicec, DIMX*DIMZ*sizeof(float));

    cudaMemcpy(deviceA, hostA, DIMX*DIMY*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, DIMY*DIMZ*sizeof(float), cudaMemcpyHostToDevice);

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , devicec , ARows , Acols, Acols ,Bcols , cRows , ccols);

    cudaMemcpy(hostc, devicec, DIMX*DIMZ*sizeof(float), cudaMemcpyDeviceToHost);
    printArray(hostA, DIMX, DIMY, 1);
    printArray(hostB, DIMY, DIMZ, 1);
    printArray(hostc, DIMX, DIMZ, 1);

    return 0;
}


void printArray(float *arr, int rows, int cols, int shouldPrint){
    if (!shouldPrint)
       return;
           
    int i,j;
 
    for(i=0; i<rows; i++){
       for(j=0; j<cols; j++){
       
          printf("%5.02f ", arr[i*cols + j]);
       }
       printf("\n");
    }
 
    printf("\n");
 }