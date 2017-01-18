# Usage:
# $ env CC=icc CXX=icpc make VAR=VAL tapas
#
# NOTE: "make CC=icc CXX=icpc ..." does not work.

.ALWAYS:

include ../Makefile.include

#LFLAGS	+= -DMASS # Use all positive sources
#LFLAGS	+= -DIneJ # Different target and source points
LFLAGS	+= -DDIRECT # Compare with direct summation

MODE ?= release
ifeq ($(MODE),debug)
	OPT = -O2
	DEBUG = -DTAPAS_DEBUG=1 -g -fno-inline
	LOG = -DTAPAS_LOG_LEVEL=1
	LFLAGS += -DASSERT # Turns on asserttions (otherwise define an empty macro function)
else ifeq ($(MODE),debug2)
	OPT = -O0
	DEBUG = -DTAPAS_DEBUG=1 -DTAPAS_DEBUG_DUMP -g
	LOG = -DTAPAS_LOG_LEVEL=1
	LFLAGS += -DASSERT # Turns on asserttions (otherwise define an empty macro function)
else
	OPT = -O3 # -O3 is also OK, but when gprof is used, gprof command misunderstand names of functions called as 'frame_dummy'
	DEBUG = -DTAPAS_DEBUG=0 -DNDEBUG
	LOG = -DTAPAS_LOG_LEVEL=0
endif

ifneq (,$(COUNT))
	CXX += -DCOUNT
	MPICXX += -DCOUNT
endif

ifneq (,$(PAPI))
	CXX += -DPAPI -lpapi
	MPICXX += -DPAPI -lpapi
endif

ifneq (,$(USE_ONESIDE_LET))
	CXX += -DTAPAS_ONESIDE_LET
	MPICXX += -DTAPAS_ONESIDE_LET
endif

#CXX += -DTAPAS_DEBUG_HISTOGRAM
#MPICXX += -DTAPAS_DEBUG_HISTOGRAM

#CXX += -DTAPAS_DEBUG_COMM_MATRIX
#MPICXX += -DTAPAS_DEBUG_COMM_MATRIX

#CXX += -DTAPAS_ONESIDE_LET
#MPICXX += -DTAPAS_ONESIDE_LET

# Test for tree construction
tree: tree.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	./a.out

# Test for serial FMM
serial: serial.o $(OBJECTS)
	$(CXX) $? $(LFLAGS) -o $@

# Test for parallel FMM
parallel: parallel.cxx $(OBJECTS)
	$(MPICXX) $^ $(LFLAGS) -o $@
#	mpirun -np 9 ./a.out --ncrit 256 --distribution plummer

tapas: tapas.d parallel_tapas parallel_tapas_mutual

# NOTE: Single node HOT code is no longer maintained.
serial_tapas: .ALWAYS $(OBJECTS) #tapas_exafmm.cxx $(OBJECTS)
	$(CXX) $(DEBUG) $(OPT) $(LOG) tapas_exafmm.cxx $(OBJECTS) $(LFLAGS) -o $@

parallel_tapas: tapas_exafmm.cxx
	$(MPICXX) $(DEBUG) $(OPT) $(LOG) -DUSE_MPI tapas_exafmm.cxx $(LFLAGS) -o $@

parallel_tapas_mutual: tapas_exafmm.cxx
	$(MPICXX) $(DEBUG) $(OPT) $(LOG) -DUSE_MPI -DFMM_MUTUAL tapas_exafmm.cxx $(LFLAGS) -o $@

tapas.d:
	$(MPICXX) -DUSE_MPI -M -MQ parallel_tapas -MQ parallel_tapas_mutual tapas_exafmm.cxx $(LFLAGS) -o $@

# Checking O(N) complexity
complexity: serial.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	for N in 10000 20000 50000 100000 200000 500000 1000000; do \
        ./a.out --numBodies $$N; echo; \
	done

# Test for kernels only
kernel: kernel.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	./a.out

# Checking O(theta^p) convergence (Turn on VTK to see a graph at the end)
convergence: kernel.cxx $(SOURCES)
	rm -f kernel.dat
	for P in 3 4 5 6 7 8 9 10 11 12 13 14 15; do echo P = $$P compiling && \
	$(CXX) $? $(LFLAGS) -DEXPANSION=$$P && \
	echo P = $$P executing && ./a.out; done

# Ewald vs. periodic FMM
ewald: ewald.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	mpirun -np 2 ./a.out --numBodies 1000 --ncrit 32 --images 3

clean:
	rm -rf *.o *.dSYM
	find . -maxdepth 1 -exec file {} \; | grep -E "(Mach-O|ELF) .* executable" | cut -d ":" -f 1 | xargs rm -f

-include tapas.d

