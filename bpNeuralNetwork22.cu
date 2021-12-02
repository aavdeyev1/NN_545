// Author: Tony Yun Tian at CSEE of EWU
// All rights are reserved!
// Please do not post this code on the Internet.

#include <iostream>
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
//#include <conio.h>


#define numIn_ 2
#define numH_ 3
#define numOut_ 1



using namespace std;

__global__ void cuda_NN(int * trainData, 
						int *trueOut,
						const int numTrainSample,
						const float learnRate,
						const long maxNumTrainIterate,
						int numNeuronIn_,
						int *indata_,
						int numNeuronHidden_,
						float h,
						float * wHidden_,
						int* vHidden_,
						int numNeuronOut_,
						float * wOut_,
						float* vOut_,
						float * yError,
						float * hError,
						float *result){
							
	int y = 0;
	h = 0;
	for(int iterate = 1; iterate <= maxNumTrainIterate; iterate ++){
		
		for(int i = 0; i < numTrainSample; i++)
			{
				for(int k = 0; k < numNeuronIn_; k++)
					//i is like row, k is like col
					//calc format is: [row * numCols + col] is same as [row][col]
					indata_[k] = trainData[i * 2 + k];

				// forward computing
				//
				//
				// compute vHidden
				for(int m = 0; m < numNeuronHidden_; m++) 
				{
					for(int k = 0; k < numNeuronIn_; k++){
						h = h + indata_[k] * wHidden_[m * 3 + (k + 1)];
					}

					h = h + wHidden_[m * 3];
					vHidden_[m] = (float)(1.0f / (1 + exp(h * (-1))));

					h = 0;
				}
			
				// compute vOut
				for(int m = 0; m < numNeuronOut_; m++)
				{
					for(int k = 0; k < numNeuronHidden_; k++)
						y = y + vHidden_[k] * wOut_[m * 4 + (k + 1)];
					y = y + wOut_[m * 4];
					vOut_[m] = (float)(1.0f / (1 + exp(h * (-1))));

					y = 0;
				}

				//
				//
				//backward compute
			
				//compute yError
				for(int m = 0; m < numNeuronOut_; m++)
					yError[m] =  vOut_[m] * ( 1 - vOut_[m]) * (  vOut_[m] - trueOut[i * 64 + m] );
				
				//compute hError
				for(int m = 0; m < numNeuronHidden_; m++)
				{
					float temp = 0;
					for(int k = 0; k < numNeuronOut_; k ++)
						temp = temp + wOut_[k * 4 + (m + 1)] * yError[k];
					hError[m] = temp * vHidden_[m] * (1 - vHidden_[m]);

				}

				//Adjust wOut[i][0] and wOut[i][j] and wHidden_
				for(int m = 0; m < numNeuronOut_; m++)
					wOut_[m * 4] = wOut_[m * 4] - learnRate * yError[m];

				for(int m = 0; m < numNeuronOut_; m++)
					for(int k = 0; k < numNeuronHidden_; k++)
						wOut_[m * 4 + (k + 1)] = wOut_[m * 4 + (k + 1)] - learnRate * yError[m] * vHidden_[k];

				for(int m = 0; m < numNeuronHidden_; m++)
				{
					wHidden_[m * 3] = wHidden_[m * 3] - learnRate * hError[m];
					for(int k = 0; k < numNeuronIn_; k++)
						wHidden_[m * 3 + (k + 1)] = wHidden_[m * 3 + (k + 1)] - learnRate * hError[m] * indata_[k];
				}
				
				//one statement below did not consider the general neural network constructure, just for this assignment
				//result[static_cast<int>(indata_[0])][static_cast<int>(indata_[1])] = vOut_[0];
				result[static_cast<int>(indata_[0]) * 8 + static_cast<int>(indata_[1])] = vOut_[0];
			/*	
			*/
			}// end for all samples
	}	

}

// template function darray_new and darray_free to allocate memory dynamically
template<class T> T** darray_new(T unit, int row, int col)
{
    int size = sizeof(T);
    void **arr = (void **) malloc(sizeof(void *) * row + size * row * col);
    if (arr != NULL)
    {
        unsigned char * head;
        head = (unsigned char *) arr + sizeof(void *) * row;
        for (int i = 0; i < row; ++i)
        {
            arr[i] =  head + size * i * col;
            for (int j = 0; j < col; ++j)
                new (head + size * (i * col + j)) T;
        }
    }
    return (T**) arr;
}

template<class T> void darray_free(T **arr, int row, int col)
{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            arr[i][j].~T();
    if (arr != NULL)
        free((void **)arr);
} 



//Implement the BP neural network

template <class T> class bpNeuralNetwork
{
	private:
      
	  //
      int numNeuronIn_;
	  int numNeuronHidden_;
	  int numNeuronOut_;

	  T indata_[numIn_];								//the input data for the input layer
	  float wHidden_[numH_][numIn_ + 1];			                //the weight belongs to Hidden Layer
	  float wOut_[numOut_][numH_ + 1];							//the weight belongs to Output Layer
	  float vHidden_[numH_];							//the value of Neuron in Hidden Layer
	  float vOut_[numOut_];								//the value of Neuron in Output Layer

  
	public:
	// Constructor
	  bpNeuralNetwork(int nIn = 2, int nH = 3, int nOut = 1) : numNeuronIn_(nIn), numNeuronHidden_(nH), numNeuronOut_(nOut)
	  {
		  int i,j;
		  //wHidden_ = darray_new( (float) 1, numNeuronHidden_,numNeuronIn_ + 1 );
		  //wOut_ = darray_new( (float) 1, numNeuronOut_,numNeuronHidden_ + 1 ); 
		  //indata_ = new T[nIn];

		  // Initiate wHidden_ to random number in U(-0.5,+0.5)
		  
		  /* initialize random seed: */
		  srand ((unsigned)time(NULL));


			//assigning random weights to the hidden neurons -bf
		  for(i = 0; i < numNeuronHidden_; i++)
		  {
              wHidden_[i][0] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
			  for(j = 1; j < numNeuronIn_ + 1; j++)
				  wHidden_[i][j] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
		  }

		  // Initiate wOut_ to random number in U(-0.5,+0.5)
		  for(i = 0; i < numNeuronOut_; i++)
		  {
              wOut_[i][0] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
			  for(j = 1; j < numNeuronHidden_ + 1; j++)
				  wOut_[i][j] = static_cast<float>((rand() % 10000 + 1 - 5000)) / 10000.0f;
		  }

		  // Initiate indata_
		  //indata_ = new T[numNeuronIn_];

		  //Initiate vHidden_
		  //vHidden_ = new float[numNeuronHidden_];

		  //Initiate vOut_
		  //vOut_ = new float[numNeuronOut_];

	  }


	// The copy constructor will be added later
	bpNeuralNetwork(const bpNeuralNetwork& initBP)
	{}


	// Destructor
	~bpNeuralNetwork()
	{
		
	}

	// Training the bpNeuralNetwork
	void training(T trainData[64][numIn_],int trueOut[64][numOut_],const int numTrainSample,const float learnRate,const long maxNumTrainIterate,float (*pLogisticFun)(float))
	{
		// row number of the trainData is the amounts of training samples, the column of the trainData  that is from column 0 to numNeuronIn_ - 1 will
		// be assigned to indata_ .
		// pointer of pLogisticFun, is a function pointer, that enable us to use other logistic function in training conveniently
		// number of rows of trueOut is equal to trainData's row number;One trueOut row corresponds to one trainData row. 
		long iterate = 0L;
		int k,m;
		float h = 0;
		//float y = 0;
		//float temp = 0;
		float* yError = new float[numNeuronOut_];
		float* hError = new float[numNeuronHidden_];
		int numE = 0;
		int width = 6;

		float mytrim(float);

		float result[64];     //Exclusively for this Assignment.The temporary matrix to store result 
                                      // converted into matrix format, in order to output more convinietly

		//Initiate the bpNetwork

		int* h_trainData;
		h_trainData = (int *)malloc(64 * numIn_ * sizeof(int));
/*
		cout << endl << "print the trainData before linearizing" << endl;
		for(int i = 0; i < 64; i++){
			for(int j = 0; j < numIn_; j++){
				cout << trainData[i][j] << " ";
			}
		}
		cout << endl;
*/
		//linearize 2d arrays
		//
		//formula is:
		//arr[(numCols * i) + j] = arr[i][j]
		for(int i = 0; i < 64; i++){
			for(int j = 0; j < numIn_; j++){
				h_trainData[(numIn_ * i) + j] = trainData[i][j];
			}
		}

		float* h_wHidden;
		h_wHidden = (float *)malloc(9 * sizeof(float));

		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				h_wHidden[(3 * i) + j] = wHidden_[i][j];
			}
		}

		float* h_wOut;
		h_wOut = (float *)malloc(numOut_ * (numH_ + 1) * sizeof(float));

		for(int i = 0; i < numOut_; i++){
			for(int j = 0; j < (numH_ + 1); j++){
				h_wOut[((numH_ + 1) * i) + j] = wOut_[i][j];
			}
		}

		float* h_trueOut; //64 rows 1 col
		h_trueOut = (float *)malloc(64 * sizeof(float));

		for(int i = 0; i < 64; i++){
			for(int j = 0; j < 1; j++){
				h_trueOut[(1 * i) + j] = trueOut[i][j];
			}
		}
/*
		cout << endl << "print the traindata after linearizing" << endl;
		for(int i = 0; i < 64 * numIn_; i++){
			
				cout << h_trainData[i] << " ";
			
		}
		cout << endl;
*/
		//setup cuda

		int* d_indata;
		int* d_vHidden;
		int* d_trainData;
		float* d_wHidden;
		float* d_h;
		float* d_vOut;
		float* d_yError;
		float* d_hError;
		float* d_wOut;
		float* d_result;
		int* d_trueOut;

		
		
		//setup block and grid size
		int blockSize, gridSize;
		blockSize = 1;
		gridSize = 1;

		//allocate memory on GPU
		cudaMalloc(&d_indata, numIn_ * sizeof(int));
		cudaMalloc(&d_trainData, 128 * sizeof(int));
		cudaMalloc(&d_trueOut, 64 * sizeof(int));
		//cudaMalloc(&d_wHidden, numH_ * sizeof(float));
		cudaMalloc(&d_h, sizeof(float));
		cudaMalloc(&d_vHidden, numH_ * sizeof(int));
		cudaMalloc(&d_vOut, numOut_ * sizeof(float));
		cudaMalloc(&d_yError, numNeuronOut_ * sizeof(float));
		cudaMalloc(&d_hError, numNeuronHidden_ * sizeof(float));
		cudaMalloc(&d_wOut, numOut_ * (numH_ + 1) * sizeof(float));
		cudaMalloc(&d_wHidden, numH_ * (numIn_ + 1) * sizeof(float));
		cudaMalloc(&d_result, 64 * sizeof(float));


		cudaMemcpy(d_indata, indata_, numIn_ * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_trainData, trainData, 128 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_trueOut, h_trueOut, 64 * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_wHidden, wHidden_, numH_ * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);	
		cudaMemcpy(d_vHidden, vHidden_, numH_ * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vOut, vOut_, numOut_ * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_yError, yError, numNeuronOut_ * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hError, hError, numNeuronHidden_ * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_wOut, h_wOut, numOut_ * (numH_ + 1) * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_wHidden, h_wHidden, numH_ * (numIn_ + 1) * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_result, result, 64 * sizeof(float), cudaMemcpyHostToDevice);
/*
		cuda_NN<<<gridSize, blockSize>>>(d_trainData, 
											d_trueOut, 
											numTrainSample, 
											learnRate, 
											maxNumTrainIterate, 
											numNeuronIn_, 
											d_indata, 
											numNeuronHidden_, 
											h, 
											d_wHidden, 
											d_vHidden, 
											numNeuronOut_, 
											d_wOut, 
											d_vOut, 
											d_yError, 
											d_hError, 
											d_result);

		cudaMemcpy(result, d_result, 64 * sizeof(float), cudaMemcpyDeviceToHost);

		cout << endl << "print the result data" << endl;
		for(int i = 0; i < 64; i++){
			
				cout << result[i] << " ";
			
		}
*/
		/*
		things i need to copy to GPU memory:
		indata_		x
		h			x
		vHidden_	x
		
		vOut_		x
		yError		x
		hError		x
		wOut_		x
		wHidden_	x
		result		x


		*/

		

		//call cuda kernel
		//cuda_NN<<<gridSize, blockSize>>>((int**)trainData, (int**)trueOut, numTrainSample, learnRate, maxNumTrainIterate, numNeuronIn_, d_indata, numNeuronHidden_, h, d_wHidden, d_vHidden, numNeuronOut_, d_wOut, d_vOut, d_yError, d_hError, d_result);

		//cudaMemcpy(result, d_result, 64 * sizeof(float), cudaMemcpyDeviceToHost);
		
		//for(iterate = 1; iterate <= maxNumTrainIterate; iterate ++)
		//{

			
			cuda_NN<<<gridSize, blockSize>>>(d_trainData, 
											d_trueOut, 
											numTrainSample, 
											learnRate, 
											maxNumTrainIterate, 
											numNeuronIn_, 
											d_indata, 
											numNeuronHidden_, 
											h, 
											d_wHidden, 
											d_vHidden, 
											numNeuronOut_, 
											d_wOut, 
											d_vOut, 
											d_yError, 
											d_hError, 
											d_result);
			/* 
			for(i = 0; i < numTrainSample; i++)
			{
				for(k = 0; k < numNeuronIn_; k++)
					indata_[k] = trainData[i][k];
				
				
				

				

				
				//create numNeuronIn_ # of threads

				// forward computing
				//
				//
				// compute vHidden
				for(m = 0; m < numNeuronHidden_; m++) 
				{
					for(k = 0; k < numNeuronIn_; k++){
						h = h + indata_[k] * wHidden_[m][k + 1];
					}

					h = h + wHidden_[m][0];
					vHidden_[m] = pLogisticFun(static_cast<float>(h));

					h = 0;
				}

				// compute vOut
				for(m = 0; m < numNeuronOut_; m++)
				{
					for(k = 0; k < numNeuronHidden_; k++)
						y = y + vHidden_[k] * wOut_[m][k + 1];
					y = y + wOut_[m][0];
					vOut_[m] = pLogisticFun(static_cast<float>(y));

					y = 0;
				}

				//
				//
				//backward compute

				//compute yError
				for(m = 0; m < numNeuronOut_; m++)
					yError[m] =  vOut_[m] * ( 1 - vOut_[m]) * (  vOut_[m] - trueOut[i][m] );
				
				//compute hError
				for(m = 0; m < numNeuronHidden_; m++)
				{
					temp = 0;
					for(k = 0; k < numNeuronOut_; k ++)
						temp = temp + wOut_[k][m + 1] * yError[k];
					hError[m] = temp * vHidden_[m] * (1 - vHidden_[m]);

				}

				//Adjust wOut[i][0] and wOut[i][j] and wHidden_
				for(m = 0; m < numNeuronOut_; m++)
					wOut_[m][0] = wOut_[m][0] - learnRate * yError[m];

				for(m = 0; m < numNeuronOut_; m++)
					for(k = 0; k < numNeuronHidden_; k++)
                        wOut_[m][k + 1] = wOut_[m][k + 1] - learnRate * yError[m] * vHidden_[k];

				for(m = 0; m < numNeuronHidden_; m++)
				{
					wHidden_[m][0] = wHidden_[m][0] - learnRate * hError[m];
					for(k = 0; k < numNeuronIn_; k++)
						wHidden_[m][k + 1] = wHidden_[m][k + 1] - learnRate * hError[m] * indata_[k];
				}
				
				//one statement below did not consider the general neural network constructure, just for this assignment
				result[static_cast<int>(indata_[0])][static_cast<int>(indata_[1])] = vOut_[0];
			
			}// end for all samples
 
			
			//output
			
			if(iterate == 10 || iterate == 100 || iterate == 1000 || iterate == 10000 || iterate == 100000)
			{
				cudaMemcpy(result, d_result, 64 * sizeof(float), cudaMemcpyDeviceToHost);
				cout << "\n\nOuput values after " << iterate << " iterations: \n";
				for(m = 0; m < 8; m++)
				{
					for(k = 0; k < 8; k ++)
					/*
						if ( (int)(result[m][k] + 0.5) == trueOut[m * 8 + k][0])
						{
							cout << setw(width) << mytrim(result[m][k]) << "  ";
						}
						else
						{
							cout << setw(width) << mytrim(result[m][k]) << "* ";
							numE ++;
						}
						
					cout << result[m][k] << " ";
					cout << "\n";
		
				}
				cout << "==> " << numE << "  errors";
				numE = 0;
                
			} // 
			*/
			

		//} // end for iteration

		cudaMemcpy(result, d_result, 64 * sizeof(float), cudaMemcpyDeviceToHost);

		

		cout << "\n\nOuput values after " << iterate << " iterations: \n";
				for(m = 0; m < 8; m++)
				{
					for(k = 0; k < 8; k ++)
						if ( (int)(result[m * 8 + k] + 0.5) == trueOut[m * 8 + k][0])
						{
							cout << setw(width) << mytrim(result[m * 8 + k]) << "  ";
						}
						else
						{
							cout << setw(width) << mytrim(result[m * 8 + k]) << "* ";
							numE ++;
						}
					cout << "\n";
		
				}
				cout << "==> " << numE << "  errors";
				numE = 0;

		cudaFree(d_indata);
		cudaFree(d_vHidden);
		cudaFree( d_wHidden);
		cudaFree(d_h);
		cudaFree(d_vOut);
		cudaFree(d_yError);
		cudaFree(d_hError);
		cudaFree( d_wOut);
		cudaFree( d_result);
		
	}// end for training

	// Classify data using the trained network
	void classifybp()
	{}  
};




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
	float fx(float);
	int i,j,k = 0,input[64][2] = {0},output[64][1] = {0};

	for(i = 0; i < 8; i ++)
		for(j = 0; j< 8; j++)
		{
			input[k][0] = i;
			input[k][1] = j;
			output[k][0] = indata[i][j];
			k ++;
		}


	bpNeuralNetwork<int> myBPNN;
	myBPNN.training( input,output,64,0.02f,100000l,fx);
	cout << "\n\n\n                Press any key to exit!";
	getchar();
	return 0;
}


//the transfer function used by neural network
float fx(float x)
{
	return (float)(1.0f / (1 + exp(x * (-1))));
}

// mytrim to make the result a precision of 3 digit
float mytrim(float x)
{
	int a = 0;
	a = static_cast<int>(x * 1000 + 0.5);      // keep a precision of 3 digit
	return (static_cast<float>(a) / 1000);
}









		


