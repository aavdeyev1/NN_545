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

__device__ float fxGPU(float *x, int idx);
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

__global__ void kernel( int *input, float *output, float *vHidden, float *wHidden, float *vOut, float *wOut, float *hError, float *yError, int numIn, int numH, int numOut, int numLayers, int numPairs )
{// Done
    // need 2D indexing for input and 3D for wHidden
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*gridDim.x + ix;
    // if(ix > numTrainSample) return;

    printf("Block: %d | Thread: %d | ix: %d\n", blockIdx.x, threadIdx.x, idx);
    // for (int q=0; q<numTrainSample*numIn;q++)
    //     printf("%5d ", input[q]);
    // printf("\n");
    int h_offset = 0 + idx;
    int y_offset = numPairs + idx;
    int temp_offset = 2*numPairs + idx;
    extern __shared__ float sums[];
    int i,j,k;
    int cols = numIn + 1;
    int rows = numH;

    sums[h_offset] = 0;
    for (k=0; k<numLayers; k++) { //2x z-dim
		for(i=0; i<rows; i++){ //3x rows
			for(j=0; j<numIn; j++){ //2x, for each w1 w2 cols
                // printf("||?%5d *%5.02f||\n", input[idx*numIn+j], wHidden[k*cols*rows + i*cols + (j+1)]);
                sums[h_offset] = sums[h_offset] + input[idx*numIn+j] * wHidden[k*cols*rows + i*cols + (j+1)];
            }
            // adding the bias weight w0
            sums[h_offset] = sums[h_offset] + wHidden[k*cols*rows + i*cols + 0];
            vHidden[idx*numH+i] = fxGPU(sums, h_offset);
            // printf("%5.02f ", sums[idx]);
            sums[h_offset] = 0;
        }
        sums[h_offset] = 0;
    }

    // y (y = second half of sums) 0-63 is for h, 64-127 is for y, 128-... is for temp
    sums[y_offset] = 0;
    rows = numOut;
    cols = numH;
    int numLongLayers = 1;
    // Compute vOut
    for (k=0; k<numLongLayers; k++) { //1x z-dim
		for(i=0; i<rows; i++){ //1x for numout
            for(j=0; j<numH; j++){ //3x, for each w1 w2 w3 cols (3hidden)
                // printf("||?%5d *%5.02f||\n", vHidden[idx*numH+j], wOut[k*cols*rows + i*cols + (j+1)]);
                sums[y_offset] = sums[y_offset] + vHidden[idx*numH+j] * wOut[k*cols*rows + i*cols + (j+1)];
            }
            // adding the bias weight w0
            sums[y_offset] = sums[y_offset] + wOut[k*cols*rows + i*cols + 0];
            vOut[idx*rows+i] = fxGPU(sums, y_offset);
            printf("%5.02f ", sums[y_offset]);
            sums[y_offset] = 0;
        }
    }
    // compute yErr
    for(i = 0; i < numOut; i++) {
        yError[idx*numOut+i] =  vOut[idx*numOut+i] * ( 1 - vOut[idx*numOut+i]) * (  vOut[idx*numOut+i] - output[idx*numOut+i] );
    }
    sums[temp_offset] = 0;
    // compute hErr
    for (k=0; k<numLongLayers; k++) { //for future z dim is num layers
        for(j = 0; j < numH; j++) { // j is for cols, numH
            sums[temp_offset] = 0;
            for(i = 0; i < numOut; i++) { // i is for rows, 1x for numOut
                // wOut -> [wbias, w1, w2, w3]xnumOut, doing [w1-w3] now
                sums[temp_offset] = sums[temp_offset] + wOut[k*cols*rows + i*cols + (j+1)] * yError[idx*numOut+i];
                // yError[idx*numOut+i] =  vOut[idx*numOut+i] * ( 1 - vOut[idx*numOut+i]) * (  vOut[idx*numOut+i] - output[idx*numOut+i] );
            }
            // printf("vHidden: %f | wOut: %f | yErr: %f\n", vHidden[idx*numH+j], wOut[k*cols*rows + i*cols + (j+1)], yError[idx*numOut+i]);
            hError[idx*numH+j] = sums[temp_offset] * vHidden[idx*numH+j]*(1 - vHidden[idx*numH+j]);
        }
    }
    
}
__global__ void adjustWeights(float learnRate, float *vHidden, float *wHidden, float *wOut, float *hError, float *yError, int numIn, int numH, int numOut, int numLayers, int numPairs )
{// Done
    // need 2D indexing for input and 3D for wHidden
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*gridDim.x + ix;
    // if(ix > numTrainSample) return;

    int i, j, k;
    int rows = numOut;
    int cols = numH;
    int numLongLayers = 1;
    // Adjust bias weight wOut
    for (k=0; k<numLongLayers; k++) { //1x z-dim
		for(i=0; i<rows; i++){ //1x for numout
            // adjusting the bias weight w0
            wOut[k*cols*rows + i*cols + 0] = wOut[k*cols*rows + i*cols + 0] - learnRate * yError[idx*numOut + i];
            printf(">>>%5.02f \n", wOut[k*cols*rows + i*cols + 0]);
        }
    }

    // // adjust wOut general weights
    for (k=0; k<numLongLayers; k++) { //1x z-dim
		for(i=0; i<rows; i++){ //1x for numout
            for(j=0; j<numH; j++){ //3x, for each w1 w2 w3 cols (3hidden)
                wOut[k*cols*rows + i*cols + +1] = wOut[k*cols*rows + i*cols + 1] - learnRate * yError[idx*numOut + i] * vHidden[idx*numH+j];
                printf(">>>%5.02f \n", wOut[k*cols*rows + i*cols + 0]);
            }
        }
    }

    // // adjust wOut general weights
    // for(m = 0; m < numNeuronOut_; m++)
    //     for(k = 0; k < numNeuronHidden_; k++)
    //         wOut_[m][k + 1] = wOut_[m][k + 1] - learnRate * yError[m] * vHidden_[k];

    // // adjust bias weight for wHidden (outer) and wHidden weights (inner)
    // for(m = 0; m < numNeuronHidden_; m++)
    // {
    //     wHidden_[m][0] = wHidden_[m][0] - learnRate * hError[m];
    //     for(k = 0; k < numNeuronIn_; k++)
    //         wHidden_[m][k + 1] = wHidden_[m][k + 1] - learnRate * hError[m] * indata_[k];
    // }
}

//the transfer function used by neural network
__device__ float fxGPU(float *x, int idx)
{
	// return (float)(1.0f / (1 + exp((float)(x * (-1)))));
    return 1.0 * x[idx];
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
		for(i=0; i<rows; i++){
			for(j=0; j<cols; j++){
		
				printf("%5.02f ", arr[k*cols*rows + i*cols + j]);
			}
			printf("\n");
   		}
   		printf("\n");
  	}

 printf("\n");
}
