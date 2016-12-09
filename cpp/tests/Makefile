.PHONY : all clean

SRC = test_set.cpp         \
			test_morton_key.cpp  \
			test_allgatherv.cpp  \
			test_gatherv.cpp     \
			test_alltoallv.cpp   \
			test_cpptricks.cpp   \
			test_geometry.cpp    \
			test_ghost_cell.cpp  \
			test_tree.cpp

DEP = $(patsubst %.cpp,%.d,$(SRC))

TRG = $(patsubst %.cpp,%,$(SRC))

all: $(TRG)

CXXFLAGS := -std=c++14 -Wall -Wextra -I../include -Wno-sign-compare -Wno-unused-function -DUSE_MPI -Werror -g
MPICXX ?= mpicxx -cxx=$(CXX)

%.d: %.cpp
	 @set -e; $(CXX) -MM $(CXXFLAGS) $< \
                | sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
                [ -s $@ ] || rm -f $@

$(TRG): % : %.o
	$(MPICXX) $(CXXFLAGS) $< -o $@

%.o:%.cpp
	$(MPICXX) $(CXXFLAGS) $< -c -o $@


clean:
	rm -f $(TRG)

list:
	echo $(TRG)

-include $(DEP)
