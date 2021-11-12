CC=g++
INCLUDE=-I/usr/local/cuda/include -I/usr/lib/mpich/include
LIBRARY_PATH=-L/usr/local/cuda/lib64  -L/usr/local/cuda/lib -L/usr/lib/mpich/lib 
LIBRARY=-lcudart -lnccl -lmpi -pthread

TARGET=test
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $< $(INCLUDE) $(LIBRARY_PATH) $(LIBRARY)

clean:
	rm -rf *.o $(TARGET)
