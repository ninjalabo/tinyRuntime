
XCODESELECT := $(shell xcode-select -p 2>/dev/null)

CC = $(if $(XCODESELECT),clang,gcc)
CXX = $(if $(XCODESELECT),clang++,g++)

CFLAGS := -Os -Wall
# compile statically if `STATIC` is set, remember configure BLIS as static library if you use it
# TODO: static increase the size of binary, optimize and reduce binary size
CFLAGS += $(if $(STATIC),-static,)

SRC := func_common.c

# use BLIS if variable `BLIS` is set
LIBBLIS_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix blis),)
LDFLAGS := $(if $(BLIS),-lblis,)
LDFLAGS +=  $(if $(and $(filter clang,$(CC)),$(BLIS)),-I$(LIBBLIS_HOME)/include -L$(LIBBLIS_HOME)/lib,)
SRC += $(if $(BLIS),func_blis.c,func.c)

# add -lm and -fopenmp after -lblis so that symbols from the libraries are available to resolve references in BLIS
LDFLAGS += -lm

# add OpenMP flags
LDFLAGS += $(if $(filter clang,$(CC)),-lomp -Xpreprocessor -fopenmp,-fopenmp)
OPENMP_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix libomp),)
LDFLAGS += $(if $(OPENMP_HOME),-I$(OPENMP_HOME)/include -L$(OPENMP_HOME)/lib,)

# add C module for quantized value calculations based on architecture
ifeq ($(ARCH),arm)
    SRC += func_q_arm.c
else ifeq ($(ARCH),x86)
    SRC += func_q_x86.c
    CFLAGS += -mavx2 -mfma
else
    SRC += func_q.c
endif

compile: 
	$(CC) $(CFLAGS) run.c $(SRC) -o run $(LDFLAGS)
	$(CC) $(CFLAGS) runq.c $(SRC) -o runq $(LDFLAGS)

ONEDNN_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix onednn),)
ONEDNN_FLAGS += -ldnnl
ONEDNN_FLAGS += $(if $(ONEDNN_HOME),-I$(ONEDNN_HOME)/include -L$(ONEDNN_HOME)/lib,)
# FIX: remove after unified runq_static.c and runq.c
compile_static:
	$(CC) $(CFLAGS) runq_static.c func_common.c func.c func_sq_onednn.c -o runq $(LDFLAGS) $(ONEDNN_FLAGS)

clean:
	rm -f run runq

cached_ipynb = $(shell git diff --cached --name-only)
pyfiles = $(patsubst %.ipynb,%.py,$(cached_ipynb))
py:
	for x in $(cached_ipynb); do ./nbexport $$x; done
	git add $(pyfiles)

nbchk:
	rm -f *.nbconvert.ipynb
	for x in $(filter-out train.ipynb prep.ipynb compare_all.ipynb test_dataset_generator.ipynb, $(wildcard *.ipynb)); do jupyter nbconvert --execute --to notebook $$x; done

CPPUTEST_HOME := $(if $(filter clang++,$(CXX)),$(shell brew --prefix cpputest),)
LDFLAGS += $(if $(filter clang++,$(CXX)),-L$(CPPUTEST_HOME)/lib -I$(CPPUTEST_HOME)/include,)
ut: utmain.c $(CPPUTESTS)
	$(CXX) -o $@ utmain.c test_func.c $(SRC) $(CFLAGS) $(LDFLAGS) -lCppUTest -lCppUTestExt $(MACROS)

pt: test_speed.c
	$(CC) -o $@ test_speed.c $(SRC) $(CFLAGS) $(LDFLAGS)

.PHONY: clean compile py nbchk ut

