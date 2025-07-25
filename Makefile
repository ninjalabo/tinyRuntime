
XCODESELECT := $(shell xcode-select -p 2>/dev/null)

CC = $(if $(XCODESELECT),clang,gcc)
CXX = $(if $(XCODESELECT),clang++,g++)

CFLAGS := -Os -Wall

COMMON_SRC := func_common.c

# use BLIS if variable `BLAS` is ON
LIBBLIS_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix blis),)
LDFLAGS := $(if $(filter ON,$(BLAS)),-lblis,)
LDFLAGS +=  $(if $(and $(filter clang,$(CC)),$(BLAS)),-I$(LIBBLIS_HOME)/include -L$(LIBBLIS_HOME)/lib,)
FUNC_SRC := $(if $(filter ON,$(BLAS)),func_blis.c,func.c)

# include oneDNN libraries if variable `BLAS` is ON
ONEDNN_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix onednn),)
LDFLAGS += $(if $(filter ON,$(BLAS)),-ldnnl,)
LDFLAGS += $(if $(ONEDNN_HOME),-I$(ONEDNN_HOME)/include -L$(ONEDNN_HOME)/lib,)

# add -lm and -fopenmp after -lblis and -lddnl so that symbols from the libraries are available to resolve references
LDFLAGS += -lm

# add OpenMP flags
LDFLAGS += $(if $(filter clang,$(CC)),-lomp -Xpreprocessor -fopenmp,-fopenmp)
OPENMP_HOME := $(if $(filter clang,$(CC)),$(shell brew --prefix libomp),)
LDFLAGS += $(if $(OPENMP_HOME),-I$(OPENMP_HOME)/include -L$(OPENMP_HOME)/lib,)

# include quantization functions
QUANT_TYPE ?= SQ
Q_FUNC_SRC := $(if $(filter DQON,$(QUANT_TYPE)$(BLAS)),func_dq_onednn.c, \
       $(if $(filter DQ,$(QUANT_TYPE)),func_dq.c, \
       $(if $(filter SQON,$(QUANT_TYPE)$(BLAS)),func_sq_onednn.c, \
       $(if $(filter SQ,$(QUANT_TYPE)),func_sq.c,))))
CFLAGS += $(if $(filter DQ,$(QUANT_TYPE)), -DUSE_DQ_FUNC,)

# compile statically if `STATIC` is ON, remember configure BLIS and oneDNN as static library if you use it
# TODO: static increase the size of binary, optimize and reduce binary size
CFLAGS += $(if $(filter ON,$(STATIC)),-static,)
LDFLAGS += $(if $(filter ON,$(STATIC)),-lstdc++,)

# Fix func_q.c is not needed in run.c and func.c not needed in runq.c
compile:
	$(CC) $(CFLAGS) run.c $(COMMON_SRC) $(FUNC_SRC) -o run $(LDFLAGS)
	$(CC) $(CFLAGS) runq.c $(COMMON_SRC) $(Q_FUNC_SRC) -o runq $(LDFLAGS)

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
CPPUTEST_FLAGS:= -lCppUTest -lCppUTestExt
CPPUTEST_FLAGS += $(if $(filter clang++,$(CXX)),-L$(CPPUTEST_HOME)/lib -I$(CPPUTEST_HOME)/include,)

SRC := $(COMMON_SRC) $(FUNC_SRC) $(Q_FUNC_SRC)
ut: utmain.c $(CPPUTESTS)
	$(CXX) -o $@ utmain.c test_func.c $(SRC) $(CFLAGS) $(LDFLAGS) $(CPPUTEST_FLAGS)

pt: test_speed.c
	$(CC) -o $@ test_speed.c $(SRC) $(CFLAGS) $(LDFLAGS)

.PHONY: clean compile py nbchk ut

