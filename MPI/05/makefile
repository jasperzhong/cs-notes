CC=mpic++
TARGET=avg all_avg
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $<
clean:
	rm $(TARGET)
