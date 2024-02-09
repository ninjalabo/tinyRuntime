# choose compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc


all: run runq

compile: 
	$(CC) -Os run.c -lm -o run
	$(CC) -Os runq.c -lm -o runq
	
run: run.c
	$(CC) -Os $< -lm -o $@
	./$@ model.bin

runq: runq.c
	$(CC) -Os $< -lm -o $@
	./$@ modelq8.bin

clean:
	rm -f run runq

.PHONY: all clean run runq compile
