all: run runq

run: run.c
	gcc -Os $< -lm -o $@
	./$@ model.bin

runq: runq.c
	gcc -Os $< -lm -o $@
	./$@ modelq8.bin

clean:
	rm -f run runq

.PHONY: all clean
