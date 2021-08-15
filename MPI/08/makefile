CC=mpic++
TARGET=split groups
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $<
clean:
	rm $(TARGET)
