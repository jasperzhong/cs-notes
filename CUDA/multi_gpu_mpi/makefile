CC=mpic++
NVCC=nvcc
TARGET=main
NVCCFLAGS=-arch=sm_60
CCFLAGS=-lrt
INCLUDES=-I/usr/local/cuda/include
all: $(TARGET)

$(TARGET): main.cc lower_bound.o
	$(CC) -g $(INCLUDES) -o $@ $^ -L/usr/local/cuda/lib64 -lcudart $(CCFLAGS)

lower_bound.o: lower_bound.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<


clean:
	rm -rf $(TARGET) *.o
