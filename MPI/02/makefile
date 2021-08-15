CC=mpic++
TARGET=send_recv ping_pong ring
all: $(TARGET)
$(TARGET): %: %.cc
	$(CC) -o $@ $<
clean:
	rm $(TARGET)
