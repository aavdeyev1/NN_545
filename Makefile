bp: bpNeuralNetwork22.o
	nvcc -arch=sm_30 -o bp bpNeuralNetwork22.o

bpNeuralNetwork22.o: bpNeuralNetwork22.cu
	nvcc -arch=sm_30 -c bpNeuralNetwork22.cu

clean:
	rm -r *.o bp
	