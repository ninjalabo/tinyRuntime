
XCODESELECT := $(shell xcode-select -p 2>/dev/null)

CC = $(if $(XCODESELECT),clang,gcc)
CXX = $(if $(XCODESELECT),clang++,g++)

CFLAGS := -Os -Wall
LDFLAGS := -lm

SRC := func_common.c

# use BLIS if variable `BLIS` is set
LIBBLIS_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix blis),)
LDFLAGS += $(if $(BLIS),-lblis,)
LDFLAGS +=  $(if $(and $(filter clang,$(CC)),$(BLIS)),-I$(LIBBLIS_HOME)/include -L$(LIBBLIS_HOME)/lib,)
SRC += $(if $(BLIS),func_blis.c,func.c)

# use arm_neon.h if ARCH=arm
SRC += $(if $(filter arm,$(ARCH)),func_q_arm.c,func_q.c)

compile: 
	$(CC) $(CFLAGS) run.c $(SRC) -o run $(LDFLAGS)
	$(CC) $(CFLAGS) runq.c $(SRC) -o runq $(LDFLAGS)

clean:
	rm -f run runq

cached_ipynb = $(shell git diff --cached --name-only)
pyfiles = $(patsubst %.ipynb,%.py,$(cached_ipynb))
py:
	for x in $(cached_ipynb); do ./nbexport $$x; done
	git add $(pyfiles)

nbchk:
	rm -f *.nbconvert.ipynb
	for x in $(filter-out train.ipynb prep.ipynb compare_all.ipynb, $(wildcard *.ipynb)); do jupyter nbconvert --execute --to notebook $$x; done

CPPUTEST_HOME := $(if $(filter clang++,$(CXX)),$(shell brew --prefix cpputest),)
LDFLAGS += $(if $(filter clang++,$(CXX)),-L$(CPPUTEST_HOME)/lib -I$(CPPUTEST_HOME)/include,)
ut: utmain.c $(CPPUTESTS)
	$(CXX) -o $@ utmain.c test_func.c $(SRC) $(CFLAGS) $(LDFLAGS) -lCppUTest -lCppUTestExt $(MACROS)

.PHONY: clean compile py nbchk ut

