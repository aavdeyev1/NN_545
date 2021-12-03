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

__global__ void matrix_multiply_simple(float* a, float* b, float* ab, int m, int n, int k) {

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
    ab[(Row*k)+Col]=Pvalue;
}

__global__ void kernel( int *input, float *output, float *vHidden, float *wHidden, int numIn, int numH, int numLayers, int numTrainSample)
{ // Done
    // need 2D indexing for input and 3D for wHidden
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*gridDim.x + ix;
    // if(ix > numTrainSample) return;

    printf("Block: %d | Thread: %d | ix: %d", blockIdx.x, threadIdx.x, idx);
    // for (int q=0; q<numTrainSample*numIn;q++)
    //     printf("%5d ", input[q]);
    // printf("\n");
    extern __shared__ float h[];
    h[0] = 0;
    int i,j,k;
    int cols = numIn + 1;
    int rows = numH;

    for (k=0; k<numLayers; k++) { //2x z-dim
		for(i=0; i<rows; i++){ //3x rows
			for(j=0; j<numIn; j++){ //2x, for each w1 w2 cols
                atomicAdd(&h[0], input[ix*numIn] * wHidden[k*cols*rows + i*cols + (j+1)]);
				// printf("%5.02f ", arr[k*cols*rows + i*cols + j]);
			// }
    // for(int layer=0; layer < numLayers; layer++) {
    //     for(int m = 0; m < numH_; m++) {
    //         for(int k = 0; k < numIn; k++) {
    //             i*cols + j
    //             atomicAdd(&h, input[k*numIn] * wHidden[m][k + 1]);
            }
            // adding the bias weight w0
            atomicAdd(&h[0], wHidden[k*cols*rows + i*cols + 0]);
            vHidden[i] = h[0];
            printf("%5.02f ", h[0]);
            h[0] = 0;
        }
    }

    // // compute vOut
    // for(int m = 0; m < numNeuronOut_; m++)
    // {
    //     for(k = 0; k < numNeuronHidden_; k++)
    //         y = y + vHidden_[k] * wOut_[m][k + 1];
    //     y = y + wOut_[m][0];
    //     vOut_[m] = pLogisticFun(static_cast<float>(y));

    //     y = 0;
    // }
    
}

// if (ix == 0)
//         wHidden[idx] = 1;
//     else
//         wHidden[idx] = 2;

// // Indexing into wHideen, set bias=1
// int h_idx = iy*(numIn + 1) + ix;
//     if (ix == 0)
//         wHidden[h_idx] = 1;
//     else
//         wHidden[h_idx] = 2;

//the transfer function used by neural network
__device__ float fxGPU(float x)
{
	// return (float)(1.0f / (1 + exp((float)(x * (-1)))));
    return 1.2;
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

 void printArray(int *arr, int rows, int cols, int shouldPrint){
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

void printArray3D(float *arr, int rows, int cols, int pages, int sP) {
	if (!sP)
	return;
		
 	int i,j,k;

	for (k=0; k<pages; k++) {
		printf("Layer %d\n", k);
		for(i=0; i<cols; i++){
			for(j=0; j<rows; j++){
		
				printf("%5.02f ", arr[k*cols*rows + i*cols + j]);
			}
			printf("\n");
   		}
   		printf("\n");
  	}

 printf("\n");
}

// __device__ void add(float *h, float *other) {
//     atomicAdd(&h, *other);
//   }