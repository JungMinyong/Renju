CFLAGS = -O2 -Wall -Wextra -std=c99
NVCCFLAGS = -O2 -arch=sm_35

ifeq ($(USE_CUDA), 1)
	CC = nvcc
	CFLAGS = $(NVCCFLAGS)
	OBJ_FILES = main.o board.o ai.o
	VERSION = CUDA
else
	CC = gcc
	OBJ_FILES = main.o board.o
	VERSION = C
endif

all: renju

renju: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o renju $(OBJ_FILES)
	@echo "Compiled the $(VERSION) version"

main.o: main.c board.h
	$(CC) $(CFLAGS) -c main.c

board.o: board.c board.h
	$(CC) $(CFLAGS) -c board.c

ai.o: ai.cu ai.h
	$(CC) $(CFLAGS) -c ai.cu

clean:
	rm -f *.o renju

