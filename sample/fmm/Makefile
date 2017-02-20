
CXXFLAGS += -I../../include -I.

### choose kernel
EQUATION = Laplace
#EQUATION = Yukawa (not available yet)
#EQUATION = Helmholtz (not available yet)
#EQUATION = Stokes (not available yet)
CXXFLAGS += -DEQUATION=$(EQUATION)

### choose basis of multipole/local expansion
#BASIS	= Cartesian
BASIS	= Spherical
#BASIS	= Planewave (not available yet)
CXXFLAGS += -D$(BASIS)

EXPANSION ?= 10
CXXFLAGS += -DFP64
CXXFLAGS += -DEXPANSION=$(EXPANSION)

CXX ?= g++
MPICXX ?= mpicxx -cxx=$(CXX) # assume mpich-like MPI

CXX_VER=$(shell $(CXX) --version)

ifneq (,$(findstring LLVM,$(CXX_VER)))
	CXXFLAGS += -std=c++11 -Wall -Wextra -O2 -funroll-loops -fcolor-diagnostics -Wno-unused-function
else ifneq (,$(findstring g++,$(CXX)))
	GCC_INLINE=-finline-functions --param max-inline-insns-recursive=2000 --param max-inline-recursive-depth=10 --param max-inline-insns-single=1000
  GCC_WARNING=-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self \
              -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual \
              -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef \
              -Wno-unused -Wno-unused-function # -Wsign-conversion
	# Mark MPI headers as system
	GCC_SYSTEM := $(shell $(MPICXX) -show | grep -oE "[-]I(\S)+" | sed -e 's/-I//g')
	CXXFLAGS += -std=c++11 -Wall -Wextra -funroll-loops -fdiagnostics-color=auto -Wno-unused-function -fno-openmp $(GCC_INLINE) #$(GCC_WARNING) -isystem $(GCC_SYSTEM)
else ifneq (,$(findstring icpc,$(CXX)))
	# for ICC, we don't use -static option (as well as -fast which implicitly includes it) because it causes a linking error.
	INTEL_INLINE=-no-inline-max-per-routine -no-inline-max-per-compile -inline-max-total-size=3000
	CXX    += -std=c++11 -Wall -Wextra -funroll-loops -xHOST -O3 -no-prec-div -fp-model fast=2 $(INTEL_INLINE)
endif

# Profiling flags (gprof)
ifneq (,$(ENABLE_PROFILING))
	CXXFLAGS += -pg -g
endif

ifneq (,$(ENABLE_SCOREP))
  CXXFLAGS += -DUSE_SCOREP
endif

ifneq (,$(MTHREAD))
	CXXFLAGS += -DMTHREAD -I$(MYTH_DIR)/include
	LDFLAGS += -L$(MYTH_DIR)/lib -lmyth-native
endif

ifneq (,$(TBB))
	ifneq (,$(findstring icpc, $(CXX)))
		CXXFLAGS += -tbb  -DTBB # for ICC, we don't use -static option (as well as -fast which implicitly includes it) because it causes a linking error.
	else
		CXXFLAGS += -DTBB -DTO_TBB -I$(TBB_DIR)/include 
		LDFLAGS	+= -L$(TBB_DIR)/lib/intel64/gcc4.4/ -L$(TBB_DIR)/build/*_release/ -ltbb
	endif
endif

MODE ?= release

ifeq ($(MODE),debug)
	CXXFLAGS += -O0 -DTAPAS_DEBUG=1 -g -gdwarf-2 -DTAPAS_LOG_LEVEL=1
else
	CXXFLAGS += -O3 -DTAPAS_DEBUG=0 -DTAPAS_LOG_LEVEL=0
endif

ifneq (,$(COUNT))
	CXXFLAGS += -DCOUNT
endif

ifneq (,$(PAPI))
	CXXFLAGS += -DPAPI
	LDFLAGS += -lpapi
endif

ifeq (1,$(USE_TWOSIDE_LET))
	CXXFLAGS += -DTAPAS_TWOSIDE_LET
else ifeq (1,$(USE_TARGET_SIDE_LET))
	CXXFLAGS += -DTAPAS_TARGET_SIDE_LET
endif


all: tapas

tapas: parallel_tapas parallel_tapas_mutual

parallel_tapas: tapas_exafmm.cxx
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI tapas_exafmm.cxx $(LDFLAGS) -o $@

parallel_tapas_mutual: tapas_exafmm.cxx
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -DFMM_MUTUAL tapas_exafmm.cxx $(LDFLAGS) -o $@

tapas.d:
	$(MPICXX) -DUSE_MPI $(CXXFLAGS) -M -MQ parallel_tapas -MQ parallel_tapas_mutual tapas_exafmm.cxx $(LDFLAGS) -o $@

clean:
	rm -rf *.o *.dSYM
	rm -rf *.d
	find . -maxdepth 1 -exec file {} \; | grep -E "(Mach-O|ELF) .* executable" | grep parallel_tapas | cut -d ":" -f 1 | xargs rm -f

-include tapas.d

