
XCODESELECT := $(shell xcode-select -p 2>/dev/null)

CC = $(if $(XCODESELECT),clang,gcc)
CXX = $(if $(XCODESELECT),clang++,g++)

CFLAGS := -Os -Wall -Wno-unused-function
LDFLAGS := -lm

SRC := func_common.c

# use BLIS if variable BLIS=1
ifeq ($(BLIS),1)
    LDFLAGS += -lblis
    SRC += func_blis.c
    ifeq ($(CC),clang)
        LIBBLIS_HOME = $(shell brew --prefix blis)
        LDFLAGS += -I$(LIBBLIS_HOME)/include -L$(LIBBLIS_HOME)/lib
    endif
else
    SRC += func.c
endif

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

ifeq ($(CXX),clang++)
    CPPUTEST_HOME := $(shell brew --prefix cpputest)
    LDFLAGS += -L$(CPPUTEST_HOME)/lib -I$(CPPUTEST_HOME)/include
endif

ut: utmain.c $(CPPUTESTS)
	$(CXX) -o $@ utmain.c test_func.c $(SRC) $(CFLAGS) $(LDFLAGS) -lCppUTest -lCppUTestExt $(MACROS)

.PHONY: clean compile py nbchk ut

