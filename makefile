# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = kernels.o 
OBJS = cpu.o

# make and compile
reduce:$(OBJS) $(GPUOBJS)
	$(NVCC) -arch=sm_52 -o NN $(OBJS) $(GPUOBJS)

kernels.o: kernels.cu
	$(NVCC) -arch=sm_52 -c kernels.cu 

cpu.o: cpu.cpp
	$(CXX) -c cpu.cpp

clean:
	rm -f *.o
	rm -f reduce
