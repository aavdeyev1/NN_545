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
// b (n * k)
// c (m * k)

__global__ void MatMulNoShared(float* a, float* b, float* ab, int m, int n, int k) {

    int Row = blockIdx.y*blockDim.y+threadIdx.y;

    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row > m) || (Col > k)) return;

    float Pvalue = 0;
    for (int i = 0; i < (TILE_DIM + n - 1)/TILE_DIM; i++) {

        for (int p = 0; p < TILE_DIM; ++p) {
            if ((i*TILE_DIM + p < n && Row < m) && (i*TILE_DIM + p < n && Col < k))
                Pvalue += a[Row*n + i*TILE_DIM + p] * b[(i*TILE_DIM + p)*k + Col];
        }
    }
    c[(Row*m)+Col]=Pvalue;
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

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , devicec , ARows , Acols, Bcols);

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