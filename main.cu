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
#define numTLayers 1 // HIDDEN LAYER W/O OUTPUT LAYER

// #define numTrainSample_ 64
#define numTrainSample_ 4

void training(int *trainData, int *trueOut, const int numTrainSample,const float learnRate,const long maxNumTrainIterate);

int main()
{
	int x = 2, y=2;
	int indata[2][2] = {
		{ 1,1},
		{ 1,0}
	};
    // int indata[y][x] = {
    //                         { 1,1,1,1, 1,1,1,1},
    //                         { 1,1,1,1, 1,1,1,1},
    //                         { 1,1,0,1, 1,1,1,1},
    //                         { 1,0,0,0, 1,1,1,1},
    //                         { 1,0,0,0, 0,0,1,1},
    //                         { 0,0,0,0, 0,0,1,1},
    //                         { 0,0,0,0, 0,1,1,1},
    //                         { 0,0,0,1, 1,1,1,1}
    //                     };
    __device__ float fxGPU(float); // init activation fn

    // Need linearized input/output for GPU.
    int i,j,k = 0,input[numIn_*numTrainSample_] = {0},output[numOut_*numTrainSample_] = {0};

    for(i = 0; i < x; i ++)
        for(j = 0; j< y; j++)
        {
            // k = rowNum
            input[k*numIn_] = i;
            input[k*numIn_ + 1] = j;
            output[k] = indata[i][j];
            k ++;
        }

    printArray(input, numTrainSample_, numIn_, 1);
    printArray(output, 1, numTrainSample_, 1);

    training(input, output, numTrainSample_,0.02f,1l);


    // bpNeuralNetwork<int> myBPNN;
    // myBPNN.training( input,output,64,0.02f,100000l,fx);
    cout << "\n\n\n                Press any key to exit!";
    getchar();
    return 0;
}


// Make cudaMemcpy and cudaMalloc to allocate memory for gpu
// Input will be the input and output arrays calculated in main numTrainSample, learnRate, long maxNumTrainIterate, *pLogisticFun
void training(int *trainData, int *trueOut, const int numTrainSample,const float learnRate,const long maxNumTrainIterate)
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

        // int* d_indata;
		// float* d_vHidden;
		// float* h_wHidden;
		float* d_h;
		// float* d_vOut;
		// float* d_yError;
		// float* d_hError;
		// float* d_wOut;
		float* d_result;

		
		
		//setup block and grid size
		int blockSize, gridSize;
		blockSize = 1;
		gridSize = 1;

        float* testW = (float *)malloc(numTLayers*numH_*(numIn_ + 1)*sizeof(float));
		float* h_W = (float *)malloc(numTLayers*numH_*(numIn_ + 1)*sizeof(float));
		testW[0] = 0.0;
		testW[1] = 1.0;
		testW[2] = 2.0;
		testW[3] = 3.0;
		testW[4] = 4.0;
		testW[5] = 5.0;
		testW[6] = 6.0;
		testW[7] = 7.0;
		testW[8] = 8.0;

		float *wOutTestIn = (float *)malloc(numOut_*(numH_+1)*sizeof(float));
		wOutTestIn[0] = 0.0;
		wOutTestIn[1] = 1.0;
		wOutTestIn[2] = 2.0;
		wOutTestIn[3] = 3.0;
		// testW[9] = 91.0;
		// testW[10] = 92.0;
		// testW[11] = 93.0;
		// testW[12] = 94.0;
		// testW[13] = 95.0;
		// testW[14] = 96.0;
		// testW[15] = 97.0;
		// testW[16] = 98.0;
		// testW[17] = 99.0;
		printf("wHidden:\n");
		printArray3D(testW, numH_, numIn_ + 1, numTLayers, 1);
		printf("wOut:\n");
		printArray3D(wOutTestIn, numOut_, numH_ + 1, 1, 1);

		// printArray3D(testW, , 4, 1, 1); // cols, rows, 

        // Allocate host mem
        int *h_input=0;
        float *h_output=0;
		float *h_vHidden=0;
		float *h_wHidden=0;
		float *h_vOut=0;
		float *h_wOut=0;
		float *h_yError=0;
		float *h_hError=0;
        h_input = (int *)malloc(numIn_*numTrainSample_*sizeof(int));
        h_output = (float *)malloc(numOut_*numTrainSample_*sizeof(float));
        h_vHidden = (float *)malloc(numTrainSample_*numH_*sizeof(float));
		h_wHidden = (float *)malloc(numTLayers*numH_*(numIn_+1)*sizeof(float)); // 3D by Layer, numNeuron, numWeight
		h_vOut = (float *)malloc(numOut_*numTrainSample_*sizeof(float));
		h_wOut = (float *)malloc(numOut_*(numH_+1)*sizeof(float)); // 3D by Layer, numNeuron, numWeight
		h_yError = (float *)malloc(numOut_*numTrainSample_*sizeof(float));
		h_hError = (float *)malloc(numTrainSample_*numH_*sizeof(float));
		
		// d_h = (float *)malloc(); // TBD
		// float* d_vOut;
		// float* d_yError;
		// float* d_hError;
		// float* d_wOut;
		// float* d_result;
        // error vector

        // Allocate dev mem
        int *d_input=0;
		float *d_output=0;
		float *d_vHidden=0;
		float *d_wHidden=0;
		float *d_vOut=0;
		float *d_wOut=0;
		float *d_yError=0;
		float *d_hError=0;
        checkCudaErrors( cudaMalloc( &d_input, numIn_*numTrainSample_*sizeof(int) ) );
        checkCudaErrors( cudaMalloc( &d_output, numOut_*numTrainSample_*sizeof(float) ) );
        checkCudaErrors( cudaMalloc( &d_vHidden, numTrainSample_*numH_*sizeof(float) ) );
        checkCudaErrors( cudaMalloc( &d_wHidden, numTLayers*numH_*(numIn_+1)*sizeof(float) ) );
        checkCudaErrors( cudaMalloc( &d_vOut, numOut_*numTrainSample_*sizeof(float) ) );
        checkCudaErrors( cudaMalloc( &d_wOut, numOut_*(numH_+1)*sizeof(float) ) );
		checkCudaErrors( cudaMalloc( &d_yError, numOut_*numTrainSample_*sizeof(float) ) );
        checkCudaErrors( cudaMalloc( &d_hError, numTrainSample_*numH_*sizeof(float) ) );

        checkCudaErrors( cudaMemcpy( d_input, trainData, numIn_*numTrainSample_*sizeof(int), cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy( d_output, trueOut, numOut_*numTrainSample_*sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy( d_wHidden, testW, numTLayers*numH_*(numIn_+1)*sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy( d_wOut, wOutTestIn, numOut_*(numH_+1)*sizeof(float), cudaMemcpyHostToDevice) );


        dim3 grid, block;

        block.x = 4;
        grid.x  = ceil( (float)numTrainSample_ / block.x );
        // grid.y  = ceil( (float)numTrainSample_ / block.y );
        
        kernel<<<grid, block, 3*numTrainSample_*sizeof(float)>>>(d_input,
								 d_output,
								 d_vHidden,
								 d_wHidden,
								 d_vOut,
								 d_wOut,
								 d_hError,
								 d_yError,
								 numIn_,
								 numH_,
								 numOut_,
								 numTLayers,
								 numTrainSample_);
        
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();        // Get error code

		if ( err != cudaSuccess )
		{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
		}
		checkCudaErrors( cudaMemcpy( h_input, d_input, numIn_*numTrainSample_*sizeof(int), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy( h_output, d_output, numOut_*numTrainSample_*sizeof(float), cudaMemcpyDeviceToHost ) );

        checkCudaErrors( cudaMemcpy( h_W, d_wHidden, numTLayers*numH_*(numIn_ + 1)*sizeof(float), cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy( h_vHidden, d_hError, numTrainSample_*numH_*sizeof(float), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy( h_vOut, d_yError, numOut_*numTrainSample_*sizeof(float), cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy( h_wOut, d_wOut, numOut_*(numH_+1)*sizeof(float), cudaMemcpyDeviceToHost ) );
        
		printf("Input:\n");
        printArray(h_input, numTrainSample_, numIn_, 1);
        // printArray(h_output, 1, numTrainSample_, 1);
		printf("hidden weights:\n");
		printArray3D(h_W, numH_, numIn_+1, numTLayers, 1);

		printf("vHidden HERROR:\n");
		printArray(h_vHidden, numTrainSample_, numH_, 1);

		printf("out weights YERROR:\n");
		printArray3D(h_wOut, numOut_, numH_+1, 1, 1);

		printf("vOut:\n");
		printArray(h_vOut, numTrainSample_, numOut_, 1);

        free( h_input );
        free( h_output );
		free( h_vHidden );
		free(h_wHidden);
		free(h_vOut);
		free(h_wOut);

        free(testW);
		free(h_W);
		free(wOutTestIn);
        
        cudaFree( d_input );
        cudaFree( d_output );
        cudaFree( d_vHidden );
		cudaFree( d_wHidden );
		cudaFree(d_vOut);
		cudaFree(d_wOut);
        
    // for each training interation in maxNumTrainIterations

    // update weights KERNEL!

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