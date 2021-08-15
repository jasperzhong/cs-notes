CC=mpic++
TARGET=broadcast
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $<
clean:
	rm $(TARGET)
