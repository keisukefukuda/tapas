# Makefile

CUDADIR = /usr/local/cuda
#SM = -gencode arch=compute_37,code=sm_37
SM = -gencode arch=compute_35,code=sm_35

CUDAOPT = -keep -Xcicc -v -Xptxas -v \
	-Xcicc -O3 -Xptxas -O3 \
	--compiler-options -std=c++11 \
	--compiler-options -fopenmp \
	--compiler-options -Wall --compiler-options -Wextra \
	--compiler-options -Wno-unused-parameter

## --expt-relaxed-constexpr

gpu35::
	$(CUDADIR)/bin/nvcc -m64 -x cu -std c++11 -ccbin icpc -I. \
		-O3 -ftz=true -gencode arch=compute_35,code=sm_35 \
		$(CUDAOPT) nbodyexample.cc

gpu37::
	$(CUDADIR)/bin/nvcc -m64 -x cu -std c++11 -ccbin icpc -I. \
		-O3 -ftz=true -gencode arch=compute_37,code=sm_37 \
		$(CUDAOPT) nbodyexample.cc

run::
	./a.out 1 56

check::
	g++ -std=c++11 -I. check.cc

#aho::
#	grep 'Total FMM' < run.raw.gpu.txt > run.gpu.txt
#	grep 'Total FMM' < run.raw.cpu.txt > run.cpu.txt

GOMI = a_dlink.* \
	*.cubin *.fatbin *.fatbin.c \
	*.ptx *.cpp[14].ii *.cpp[23].i *.cu.cpp.ii \
	*.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c \
	*.cudafe2.c *.cudafe2.gpu *.cudafe2.stub.c \
	*.hash *.module_id

clean::
	rm -f a.out *.o core.* $(GOMI)
