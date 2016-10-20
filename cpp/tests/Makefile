.PHONY : all clean
targets = test_set test_morton_key test_allgatherv test_gatherv test_alltoallv test_cpptricks test_geometry test_ghost_cell

all: $(targets)

CXXFLAGS := -std=c++14 -Wall -Wextra -I../include -Wno-sign-compare -DUSE_MPI -Werror -g
MPICXX ?= mpicxx -cxx=$(CXX)

$(targets): % : %.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(targets)

list:
	echo $(targets)

