
XCODESELECT := $(shell xcode-select -p 2>/dev/null)

CC = $(if $(XCODESELECT),clang,gcc)
CXX = $(if $(XCODESELECT),clang++,g++)

compile: 
	$(CC) -Os -Wall run.c  func.c -lm -o run
	$(CC) -Os -Wall runq.c func.c -lm -o runq

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
    CPPUTEST_HOME += -L/opt/homebrew/opt/cpputest/lib -I/opt/homebrew/opt/cpputest/include
endif
CPPUTESTS = func.c test_func.c
ut: utmain.c $(CPPUTESTS)
	$(CXX) -o $@ utmain.c $(CPPUTESTS) $(LDFLAGS) $(CPPUTEST_HOME) -lCppUTest -lCppUTestExt

.PHONY: clean compile py nbchk ut

