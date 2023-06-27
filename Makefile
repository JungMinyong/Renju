CC = gcc
CFLAGS = -Wall -Wextra

all: renju

renju: main.o board.o
	$(CC) $(CFLAGS) -o renju main.o board.o

main.o: main.c board.h
	$(CC) $(CFLAGS) -c main.c

board.o: board.c board.h
	$(CC) $(CFLAGS) -c board.c

clean:
	rm -f *.o renju

