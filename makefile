# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = kernels.o main.o
OBJS = cpu.o

# make and compile
NN:$(OBJS) $(GPUOBJS)
	$(NVCC) -arch=sm_52 -o NN $(OBJS) $(GPUOBJS)

main.o: main.cu
	$(NVCC) -arch=sm_52 -c main.cu 

kernels.o: kernels.cu
	$(NVCC) -arch=sm_52 -c kernels.cu 

cpu.o: cpu.cpp
	$(CXX) -c cpu.cpp

clean:
	rm -f *.o
	rm -f reduce
