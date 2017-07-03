/* allpairs_cuda.h -*- Mode: C++; Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2017 RIKEN AICS */

/* Tapas All-pairs */

#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

#define BR0_ {
#define BR1_ }

#ifdef __CUDACC__

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

namespace tapas BR0_

/* The number of command streams. */

#define TAPAS_CUDA_MAX_NSTREAMS 128

/* Table of core counts on an SM by compute-capability. (There is
   likely no way to get the core count; See deviceQuery in CUDA
   samples). */

static struct TESLA_CORES {int sm; int n_cores;} tesla_cores[] = {
    {10, 8}, {11, 8}, {12, 8}, {13, 8},
    {20, 32}, {21, 48},
    {30, 192}, {32, 192}, {35, 192}, {37, 192},
    {50, 128},
};

static std::atomic<int> tesla_streamid (0);

/* GPU State of A Process.  It assumes use of a single GPU for each
   MPI process.  NSTREAMS is the number of command streams (There is
   likely no bound of the count). (32 maximum concurrent kernels on
   Kepler sm_35).  NCONNECTIONS is the number of physical command
   streams. (default=8 and maximum=32 on Kepler sm_35). */

struct TESLA {
    int gpuno;
    int sm;
    int n_sm;
    int n_cores;
    size_t scratchpad_size;
    size_t max_cta_size;
    int kernel_max_cta_size;
    /* Tapas options */
    int cta_size;
    int n_streams;
    int n_connections;
#ifdef __CUDACC__
    cudaStream_t streams[TAPAS_CUDA_MAX_NSTREAMS];
#endif
};

#if 0 /*standalone*/

static void check_error(const char *msg, const char *file, const int line) {
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
	fprintf(stderr,
		"%s:%i (%s): CUDA ERROR (%d): %s.\n",
		file, line, msg, (int)ce, cudaGetErrorString(ce));
	//cudaError_t ce1 = cudaDeviceReset();
	//assert(ce1 == cudaSuccess);
	assert(ce == cudaSuccess);
    }
}

static void setup_tesla(int cta, int nstreams) {
    assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

    tesla_device.cta_size = cta;
    tesla_device.n_streams = nstreams;

    cudaError_t ce;
    int ngpus;
    ce = cudaGetDeviceCount(&ngpus);
    assert(ce == cudaSuccess);

    tesla_device.gpuno = 0;
    cudaDeviceProp prop;
    ce = cudaGetDeviceProperties(&prop, tesla_device.gpuno);
    assert(ce == cudaSuccess);
    ce = cudaSetDevice(tesla_device.gpuno);
    assert(ce == cudaSuccess);

    assert(prop.unifiedAddressing);

    tesla_device.sm = (prop.major * 10 + prop.minor);
    tesla_device.n_sm = prop.multiProcessorCount;

    tesla_device.n_cores = 0;
    for (struct TESLA_CORES& i : tesla_cores) {
	if (i.sm == tesla_device.sm) {
	    tesla_device.n_cores = i.n_cores;
	    break;
	}
    }
    assert(tesla_device.n_cores != 0);

    tesla_device.scratchpad_size = prop.sharedMemPerBlock;
    tesla_device.max_cta_size = prop.maxThreadsPerBlock;
    assert(prop.maxThreadsPerMultiProcessor >= prop.maxThreadsPerBlock * 2);

    for (int i = 0; i < tesla_device.n_streams; i++) {
	ce = cudaStreamCreate(&tesla_device.streams[i]);
	assert(ce == cudaSuccess);
    }
}

static void release_tesla() {
    for (int i = 0; i < tesla_device.n_streams; i++) {
	cudaError_t ce = cudaStreamDestroy(tesla_device.streams[i]);
	assert(ce == cudaSuccess);
    }
}

static void tapas_start() {}

static void tapas_finish() {
    check_error("tapas_finish", __FILE__, __LINE__);
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
	fprintf(stderr,
		"%s:%i (%s): CUDA ERROR (%d): %s.\n",
		__FILE__, __LINE__, "cudaDeviceSynchronize",
		(int)ce, cudaGetErrorString(ce));
	assert(ce == cudaSuccess);
    }
}

#endif /*standalone*/

#if 0 /*GOMI*/

/* Memory allocator for GPU unified-memory.  It will replace vector
   allocators.  (MEMO: Its name should be a generic one.) */

template <typename T>
struct um_allocator : public std::allocator<T> {
public:
    /*typedef T* pointer;*/
    /*typedef const T* const_pointer;*/
    /*typedef T value_type;*/

    template <class U> struct rebind {typedef um_allocator<U> other;};

    T* allocate(size_t n, const void* hint = 0) {
        T* p;
        cudaError_t ce;
        ce = cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachGlobal);
        assert(ce == cudaSuccess && p != 0);
        fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd\n", p, n);
        fflush(0);
        return p;
    }

    void deallocate(T* p, size_t n) {
        cudaError_t ce = cudaFree(p);
        assert(ce == cudaSuccess);
        fprintf(stderr, ";; cudaFree() p=%p n=%zd\n", p, n); fflush(0);
    }

    explicit um_allocator() throw() : std::allocator<T>() {}

    explicit um_allocator(const std::allocator<T>& a) throw()
        : std::allocator<T>(a) {}

    template <class U> explicit
    um_allocator(const um_allocator<U>& a) throw()
        : std::allocator<T>(a) {}

    ~um_allocator() throw() {}
};

template <typename T>
using gpu_vector = std::vector<T, um_allocator<T>>;

#endif /*GOMI*/

/*
 | fun allpairs H G Z X Y W =
 |    map (fn (xi, wi) =>
 |            H (foldl (fn (yj, a) => (G xi yj wi) a) Z Y) wi)
 |        (zip X W)
*/

#if 0 /*AHO*/

template <class FnH, class FnG, class Z, class FnF,
          class VecR, class VecX, class VecY, class VecW>
__global__
void allpairs_kernel(FnH h, FnG g, Z z, FnF f,
                     VecR r, VecX x, VecY y, VecW w,
                     size_t xsz, size_t ysz, int tilesize) {
    /*static_assert(std::is_function<FnG>::value, "Fn g");*/
    using X = typename VecX::value_type;
    using Y = typename VecY::value_type;
    using W = typename VecW::value_type;
    //typedef typename VecR::value_type R;
    assert(tilesize <= blockDim.x);
    int index = (blockDim.x * blockIdx.x + threadIdx.x);
    extern __shared__ Y scratchpad[];
    int ntiles = TAPAS_CEILING(ysz, tilesize);

    X& xi = ((index < xsz) ? x[index] : x[0]);
    W& wi = ((index < xsz) ? w[index] : w[0]);
    Z acc;
    acc = z;
    for (int t = 0; t < ntiles; t++) {
        if ((tilesize * t + threadIdx.x) < ysz && threadIdx.x < tilesize) {
            scratchpad[threadIdx.x] = y[tilesize * t + threadIdx.x];
        }
        __syncthreads();

        if (index < xsz) {
            unsigned int jlim = min(tilesize, (int)(ysz - tilesize * t));
#pragma unroll 64
            for (unsigned int j = 0; j < jlim; j++) {
                Y& yj = scratchpad[j];
		/*acc = g(f(xi, yi), acc);*/
                g(xi, yj, wi);
            }
        }
        __syncthreads();
    }
#if 0 /*AHO*/
    if (index < xsz) {
        if (std::is_function<FnH>::value) {
            r[index] = h(wi, acc);
        }
    }
#endif
}

#else

template <class FnH, class FnG, class Z,
          class VecR, class VecX, class VecY, class VecW>
__global__
void allpairs_kernel(FnH h, FnG g, Z zero,
                     VecR r, VecX x, VecY y, VecW w,
                     size_t xsz, size_t ysz, int tilesize) {
    assert(tilesize <= blockDim.x);

    using X = typename VecX::value_type;
    using Y = typename VecY::value_type;
    using W = typename VecW::value_type;

    int index = (blockDim.x * blockIdx.x + threadIdx.x);
    int ntiles = TAPAS_CEILING(ysz, tilesize);
    extern __shared__ Y scratchpad[];

    X& xi = x[index];
    W& wi = w[index];

    Z acc;
    acc = zero;
    for (int t = 0; t < ntiles; t++) {
        if ((tilesize * t + threadIdx.x) < ysz && threadIdx.x < tilesize) {
            scratchpad[threadIdx.x] = y[tilesize * t + threadIdx.x];
        }
        __syncthreads();

        if (index < xsz) {
            unsigned int jlim = min(tilesize, (int)(ysz - tilesize * t));
#pragma unroll 64
            for (unsigned int j = 0; j < jlim; j++) {
                Y& yj = scratchpad[j];
                acc = g(xi, yj, wi, acc);
                /*g(xi, yj, wi);*/
            }
        }
        __syncthreads();
    }
    if (r != 0) {
        if (index < xsz) {
            r[index] = h(wi, acc);
        }
    }
}

#endif

template <class FnH, class FnG, class Z,
          class VecR, class VecX, class VecY, class VecW>
static void
allpairs(TESLA& tesla,
         int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
         FnH h, FnG g, Z z, VecR r, VecX x, VecY y, VecW w) {
    {
        size_t xsz = x.size();
        size_t ysz = y.size();
        assert(xsz != 0 && ysz != 0);

        /* Overwrite th eparameters. */

#if 1
        static std::mutex ap_mutex;
        static struct cudaFuncAttributes ap_tesla_attr;

        if (ap_tesla_attr.binaryVersion == 0) {
            ap_mutex.lock();
            cudaError_t ce = cudaFuncGetAttributes(
                                                   &ap_tesla_attr,
                                                   &allpairs_kernel<FnH, FnG, Z, VecR, VecX, VecY, VecW>);
            assert(ce == cudaSuccess);
            ap_mutex.unlock();
        }
        assert(ap_tesla_attr.binaryVersion != 0);

        int cta0 = (TAPAS_CEILING(tesla.cta_size, 32) * 32);
        int ctasize = std::min(cta0, ap_tesla_attr.maxThreadsPerBlock);
        assert(ctasize == tesla.cta_size);

        typedef typename VecX::value_type X;
        int tile0 = (tesla.scratchpad_size / sizeof(X));
        int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
        int tilesize = std::min(ctasize, tile1);
        assert(tilesize > 0);

        int scratchpadsize = (sizeof(X) * tilesize);
        size_t nblocks = TAPAS_CEILING(xsz, ctasize);
#endif

        //tesla_streamid = 0;
        tesla_streamid++;
        int s = (tesla_streamid % tesla.n_streams);

        /*AHO*/
        if (0) {
            printf("allpairs(nblocks=%ld ctasize=%d"
                   " scratchpadsize=%d tilesize=%d)\n",
                   nblocks, ctasize, scratchpadsize, tilesize);
            fflush(0);
        }

        allpairs_kernel<<<nblocks, ctasize, scratchpadsize,
            tesla.streams[s]>>>
            (h, g, z, r, x, y, w, xsz, ysz, tilesize);
    }
}

BR1_

//#undef TAPAS_CEILING
//#undef TAPAS_FLOOR

#else /*__CUDACC__*/
#endif /*__CUDACC__*/

#undef BR0_
#undef BR1_

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
