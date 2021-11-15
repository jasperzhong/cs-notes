CC=g++
CFLAGS=-g1
INCLUDE=-I/usr/local/cuda/include -I/usr/include/openmpi
LIBRARY_PATH=-L/usr/local/cuda/lib64  -L/usr/local/cuda/lib 
LIBRARY=-lcudart -lmpi -lmpi_cxx -pthread -ldl -lrt
STATIC_LIBRARY=/usr/local/cuda/lib/libnccl_static.a

TARGET=test
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $< $(STATIC_LIBRARY) $(CFLAGS) $(INCLUDE) $(LIBRARY_PATH) $(LIBRARY) 

clean:
	rm -rf *.o $(TARGET)
