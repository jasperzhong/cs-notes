CC=mpic++
TARGET=reduce_avg reduce_stddev
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $<
clean:
	rm $(TARGET)
