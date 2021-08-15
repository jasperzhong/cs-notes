CC=mpic++
CFLAGS=-g
TARGET=random_rank
HEADERFILE=tmpi_rank.h
SRCFILE=tmpi_rank.cc

all: $(TARGET)
$(TARGET): %: %.cc $(HEADERFILE) $(SRCFILE)
	$(CC) $(CFLAGS) -o $@ $< $(HEADERFILE) $(SRCFILE)
clean:
	rm $(TARGET)
