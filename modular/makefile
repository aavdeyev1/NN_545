CXX = g++ -g -Wall

CFLAGS += -std=c++11 -I./

objects = demo_driver.o tensor.o tensor_ops.o tests.o layer.o matrix_math.o metrics.o dataloader.o

%.o: %.cpp
	$(CXX) -c -g $(CFLAGS) $< -o $@

demo_driver: ${objects}
	g++ -o demo_driver ${objects}

clean:
	rm -f ./*.o