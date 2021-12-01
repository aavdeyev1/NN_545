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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include "cpu.h"
#include "kernels.h"

// Make static array for these numbers so we can grow the number of hidden layers
#define numIn_ 2
#define numH_ 3
#define numOut_ 1

#define numTrainSample_ 64

void printArray(int *arr, int rows, int cols, int shouldPrint);
__global__ void kernel4( int *input, int *output, int numIn, int numPairs );
void training(int *trainData, int *trueOut, const int numTrainSample,const float learnRate,const long maxNumTrainIterate,float (*pLogisticFun)(float));

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
    float fxGPU(float); // init activation fn

    // Need linearized input/output for GPU.
    int i,j,k = 0,input[numIn_*numTrainSample_] = {0},output[numOut_*numTrainSample_] = {0};

    for(i = 0; i < 8; i ++)
        for(j = 0; j< 8; j++)
        {
            // k = rowNum
            input[k*numIn_] = i;
            input[k*numIn_ + 1] = j;
            output[k] = indata[i][j];
            k ++;
        }


    printArray(input, numTrainSample_, numIn_, 1);
    printArray(output, 1, numTrainSample_, 1);

    training(input, output, numTrainSample_,0.02f,1l,fxGPU);


    // bpNeuralNetwork<int> myBPNN;
    // myBPNN.training( input,output,64,0.02f,100000l,fx);
    cout << "\n\n\n                Press any key to exit!";
    getchar();
    return 0;
}


// Make cudaMemcpy and cudaMalloc to allocate memory for gpu
// Input will be the input and output arrays calculated in main numTrainSample, learnRate, long maxNumTrainIterate, *pLogisticFun
void training(int *trainData, int *trueOut, const int numTrainSample,const float learnRate,const long maxNumTrainIterate,float (*pLogisticFun)(float))
{
    // row number of the trainData is the amounts of training samples, the column of the trainData  that is from column 0 to numNeuronIn_ - 1 will
		// be assigned to indata_ .
		// pointer of pLogisticFun, is a function pointer, that enable us to use other logistic function in training conveniently
		// number of rows of trueOut is equal to trainData's row number;One trueOut row corresponds to one trainData row. 
		long iterate = 0L;
		int i,k,m;
		float h = 0;
		float y = 0;
		float temp = 0;
		float* yError;
		float* hError;
		int numE = 0;
		int width = 6;

		float mytrim(float);

        // Allocate host mem
        int *h_input=0, *h_output=0;
        h_input = (int *)malloc(numIn_*numTrainSample_*sizeof(int));
        h_output = (int *)malloc(numOut_*numTrainSample_*sizeof(int));
        // error vector

        // Allocate dev mem
        int *d_input=0, *d_output=0;
        checkCudaErrors( cudaMalloc( &d_input, numIn_*numTrainSample_*sizeof(int) ) );
        checkCudaErrors( cudaMalloc( &d_output, numOut_*numTrainSample_*sizeof(int) ) );
        printf("%d\n", numIn_*numTrainSample_);

        checkCudaErrors( cudaMemcpy( d_input, trainData, numIn_*numTrainSample_*sizeof(int), cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy( d_output, trueOut, numOut_*numTrainSample_, cudaMemcpyHostToDevice) );

        dim3 grid, block;

        block.x = 32;
        block.y = 32;
        grid.x  = ceil( (float)numTrainSample_ / block.x );
        grid.y  = 1;
        
        kernel4<<<grid, block>>>(d_input, d_output, numIn_, numTrainSample_);
        checkCudaErrors( cudaMemcpy( h_input, d_input, numIn_*numTrainSample_, cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy( h_output, d_output, numOut_*numTrainSample_, cudaMemcpyDeviceToHost ) );

        printArray(h_input, numTrainSample_, numIn_, 1);
        printArray(h_output, 1, numTrainSample_, 1);

        free( h_input );
        free( h_output );
        cudaFree( d_input );
        cudaFree( d_output );
        
    // for each training interation in maxNumTrainIterations

    // update weights

    // for each training pair k:




        // kernel call: 1 per input
        // inputs: trainDataInput vector
        //  trueOut vector, length numOut
        //  hidden weights H

        

		
	// 	for(iterate = 1; iterate <= maxNumTrainIterate; iterate ++)
	// 	{
	// 		for(i = 0; i < numTrainSample; i++)
	// 		{
	// 			// make an input vector of len num of input data, for a given training sample i
	// 			for(k = 0; k < numIn_; k++)
	// 				indata_[k] = trainData[i][k];
				

	// 			// forward computing
	// 			//
	// 			//
	// 			// compute vHidden
	// 			for(m = 0; m < numNeuronHidden_; m++)
	// 			{
	// 				for(k = 0; k < numNeuronIn_; k++)
	// 					h = h + indata_[k] * wHidden_[m][k + 1];
	// 				h = h + wHidden_[m][0];
	// 				vHidden_[m] = pLogisticFun(static_cast<float>(h));

	// 				h = 0;
	// 			}

	// 			// compute vOut
	// 			for(m = 0; m < numNeuronOut_; m++)
	// 			{
	// 				for(k = 0; k < numNeuronHidden_; k++)
	// 					y = y + vHidden_[k] * wOut_[m][k + 1];
	// 				y = y + wOut_[m][0];
	// 				vOut_[m] = pLogisticFun(static_cast<float>(y));

	// 				y = 0;
	// 			}

	// 			//
	// 			//
	// 			//backward compute

	// 			//compute yError
	// 			for(m = 0; m < numNeuronOut_; m++)
	// 				yError[m] =  vOut_[m] * ( 1 - vOut_[m]) * (  vOut_[m] - trueOut[i][m] );
				
	// 			//compute hError
	// 			for(m = 0; m < numNeuronHidden_; m++)
	// 			{
	// 				temp = 0;
	// 				for(k = 0; k < numNeuronOut_; k ++)
	// 					temp = temp + wOut_[k][m + 1] * yError[k];
	// 				hError[m] = temp * vHidden_[m] * (1 - vHidden_[m]);

	// 			}

	// 			//Adjust wOut[i][0] and wOut[i][j] and wHidden_
	// 			for(m = 0; m < numNeuronOut_; m++)
	// 				wOut_[m][0] = wOut_[m][0] - learnRate * yError[m];

	// 			for(m = 0; m < numNeuronOut_; m++)
	// 				for(k = 0; k < numNeuronHidden_; k++)
    //                     wOut_[m][k + 1] = wOut_[m][k + 1] - learnRate * yError[m] * vHidden_[k];

	// 			for(m = 0; m < numNeuronHidden_; m++)
	// 			{
	// 				wHidden_[m][0] = wHidden_[m][0] - learnRate * hError[m];
	// 				for(k = 0; k < numNeuronIn_; k++)
	// 					wHidden_[m][k + 1] = wHidden_[m][k + 1] - learnRate * hError[m] * indata_[k];
	// 			}
				
	// 			//one statement below did not consider the general neural network constructure, just for this assignment
	// 			result[static_cast<int>(indata_[0])][static_cast<int>(indata_[1])] = vOut_[0];
			
	// 		}// end for all samples

	// 		} // 

	// 	} // end for iteration
		
	// }// end for training


}

__global__ void kernel4( int *input, int *output, int numIn, int numPairs )
{ // Done
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*numIn + ix;
    // if (ix < numPairs)
    output[idx] = 1;
    
}

//the transfer function used by neural network
__device__ float fxGPU(float x)
{
	return (float)(1.0f / (1 + exp((float)(x * (-1)))));
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