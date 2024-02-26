# choose compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

compile: 
	$(CC) -Os -Wall run.c -lm -o run
	$(CC) -Os -Wall runq.c -lm -o runq

clean:
	rm -f run runq

cached_ipynb = $(shell git diff --cached --name-only)
pyfiles = $(patsubst %.ipynb,%.py,$(cached_ipynb))
py:
	for x in $(cached_ipynb); do ./nbexport $$x; done
	git add $(pyfiles)

.PHONY: clean compile py
