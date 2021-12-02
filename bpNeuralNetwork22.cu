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
		int i,k,m;
		float h = 0;
		float y = 0;
		float temp = 0;
		float* yError = new float[numNeuronOut_];
		float* hError = new float[numNeuronHidden_];
		int numE = 0;
		int width = 6;

		float mytrim(float);

		float result[8][8] = {0};     //Exclusively for this Assignment.The temporary matrix to store result 
                                      // converted into matrix format, in order to output more convinietly

		//Initiate the bpNetwork

		//setup cuda

		//copy indata_ to GPU
		//copy wHidden_ to GPU

		int *d_a;
		int *d_c;

		//move this up
		cudaMalloc(&d_a, indata_*sizeOf(int));

		//need to linearize wHidden
		cudaMalloc(&d_c, wHidden_*sizeOf(int));

		
		for(iterate = 1; iterate <= maxNumTrainIterate; iterate ++)
		{
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
					for(k = 0; k < numNeuronIn_; k++) 
						h = h + indata_[k] * wHidden_[m][k + 1];
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
				cout << "\n\nOuput values after " << iterate << " iterations: \n";
				for(m = 0; m < 8; m++)
				{
					for(k = 0; k < 8; k ++)
						if ( (int)(result[m][k] + 0.5) == trueOut[m * 8 + k][0])
						{
							cout << setw(width) << mytrim(result[m][k]) << "  ";
						}
						else
						{
							cout << setw(width) << mytrim(result[m][k]) << "* ";
							numE ++;
						}
					cout << "\n";
		
				}
				cout << "==> " << numE << "  errors";
				numE = 0;
                
			} // 

		} // end for iteration
		
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









		


