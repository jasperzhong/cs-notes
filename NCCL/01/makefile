CC=g++
CFLAGS=-g1
INCLUDE=-I/usr/local/cuda/include -I/usr/include/openmpi
LIBRARY_PATH=-L/usr/local/cuda/lib64  -L/usr/local/cuda/lib 
LIBRARY=-lcudart -lnccl -lmpi -lmpi_cxx -pthread

TARGET=test
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $< $(CFLAGS) $(INCLUDE) $(LIBRARY_PATH) $(LIBRARY)

clean:
	rm -rf *.o $(TARGET)
