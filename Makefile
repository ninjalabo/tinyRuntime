# choose compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

compile: 
	$(CC) -Os run.c -lm -o run
	$(CC) -Os runq.c -lm -o runq

clean:
	rm -f run runq

.PHONY: clean compile
