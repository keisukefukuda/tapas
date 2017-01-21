DEPFILE=bh.d

MODE ?= release

ifeq ($(MODE),debug)
	OPT = -O0
	DEBUG = -DTAPAS_DEBUG=1 -g -gdwarf-2
	LOG = -DTAPAS_LOG_LEVEL=1
else
	OPT = -O3 # -O3 is also OK, but when gprof is used, gprof command misunderstand names of functions called as 'frame_dummy'
	DEBUG = -DTAPAS_DEBUG=0
	LOG = -DTAPAS_LOG_LEVEL=0
endif

CC ?= clang
CXX ?= clang++
MPICXX ?= mpicxx
CXXFLAGS := -std=c++11 -Wall -Wextra -fno-elide-constructors -fopenmp -Wno-unused-function
LDFLAGS := -fopenmp

ifneq (,$(USE_ONESIDE_LET))
	CXXFLAGS += -DTAPAS_ONESIDE_LET
endif

#PROF := -pg

ifneq (,$(findstring clang++,$(CXX)))
	CXXFLAGS += -funroll-loops -fcolor-diagnostics -Wno-sign-compare -Wno-unused-function
else ifneq (, $(findstring g++, $(CXX)))
	CXXFLAGS += -funroll-loops -fdiagnostics-color=auto \
							-Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-function
else ifneq (, $(findstring icpc, $(CXX)))
endif

CXXFLAGS += -DS=$(S)

TAPAS_DIR = ../../cpp

all: bh

#bh.d: bh.cc
#	$(MPICXX) -MM -DUSE_MPI -o $@ $(CXXFLAGS)  -I$(TAPAS_DIR)/include $(OPT) $(DEBUG) $(LOG) -c $< $(PROF)

bh.o: bh.cc
	$(MPICXX) -DDUMP -DUSE_MPI -o $@ $(CXXFLAGS)  -I$(TAPAS_DIR)/include $(OPT) $(DEBUG) $(LOG) -c $< $(PROF)

bh: bh.o
	$(MPICXX) -DUSE_MPI $^ $(LDFLAGS) $(PROF) -o $@ $(PROF)

bh.d: bh.cc
	$(MPICXX) -MM -DUSE_MPI -o $@ $(CXXFLAGS) -I$(TAPAS_DIR)/include $^

clean:
	$(RM) -f *.o bh bh_mpi

-include bh.d
