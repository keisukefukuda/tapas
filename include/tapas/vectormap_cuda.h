/* vectormap_cuda.h -*- Mode: C++; Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2017 RIKEN AICS */

/** @file vectormap_cuda.h @brief Direct part by CUDA.  See
    "vectormap_cpu.h". */

#pragma once

#include <type_traits>

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include "allpairs_cuda.h"
#include "tapas/vectormap_util.h"

#define BR0_ {
#define BR1_ }

namespace tapas BR0_

#define ALLPAIRS 1

#if 0 /*allpairs*/

/* Table of core counts on an SM by compute-capability. (There is
   likely no way to get the core count; See deviceQuery in CUDA
   samples). */

static struct TESLA_CORES {int sm; int n_cores;} tesla_cores[] = {
    {10, 8}, {11, 8}, {12, 8}, {13, 8},
    {20, 32}, {21, 48},
    {30, 192}, {32, 192}, {35, 192}, {37, 192},
    {50, 128},
};

static std::atomic<int> streamid (0);

/* The number of command streams. */

#define TAPAS_CUDA_MAX_NSTREAMS 128

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

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

#endif /*allpairs*/

namespace {

inline void vectormap_check_error(const char* msg, const char* file, const int line) {
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

/**
 * \breif Atomic-add code from cuda-c-programming-guide (double
 * precision).
 */

__device__
static double atomicAdd(double* address, double val) {
    // Should we use uint64_t ?
    static_assert(sizeof(unsigned long long int) == sizeof(double), "sizeof(unsigned long long int) == sizeof(double)");
    static_assert(sizeof(unsigned long long int) == sizeof(uint64_t), "sizeof(unsigned long long int) == sizeof(uint64_t)");

    unsigned long long int* address1 = (unsigned long long int*)address;
    unsigned long long int chk;
    unsigned long long int old;
    chk = *address1;
    do {
        old = chk;
        chk = atomicCAS(address1, old,
                        __double_as_longlong(val + __longlong_as_double(old)));
    } while (old != chk);
    return __longlong_as_double(old);
}

/**
 * \breif Atomic-add code from cuda-c-programming-guide (single
 * precision).
 */

__device__
static float atomicAdd(float* address, float val) {
    // Should we use uint32_t ?
    static_assert(sizeof(int) == sizeof(float), "sizeof(int) == sizeof(float)");
    static_assert(sizeof(uint32_t) == sizeof(float), "sizeof(int) == sizeof(float)");

    int* address1 = (int*)address;
    int chk;
    int old;
    chk = *address1;
    do {
        old = chk;
        chk = atomicCAS(address1, old,
                        __float_as_int(val + __int_as_float(old)));
    } while (old != chk);
    return __int_as_float(old);
}

template <class T0, class T1, class T2>
struct cellcompare_r {
    inline bool operator() (const std::tuple<T0, T1, T2>& i,
                            const std::tuple<T0, T1, T2>& j) {
        const auto* ip = std::get<2>(i).data();
        const auto* jp = std::get<2>(j).data();
        return (ip < jp);
    }
};

}

struct Cell_Item_t {int cell; int item;};

/* Array of data (Body or BodyAttr) in a cell.  The elements are
   filled contiguously across cell boundaries, but they are indexed
   with a gap to align a cell boundary to a warp boundary (rounded up
   to 32 elements at a cell boundary). */

// T is expected to be Body or BodyAttr of a certain cell
// The cell's bodies/body attritbutes starts from (base_ptr + ofst).
// Thus i-th Body/BodyAttr of the cell is
//    base_ptr[ofst + i]
// where (0 <= i < size)

template <class T>
struct Cell_Data {
    int size;
    size_t ofst;
    T* base_ptr;

    __host__ __device__ __forceinline__
    T* data() {
        return base_ptr + ofst;
    }

    __host__ __device__ __forceinline__
    const T* data() const {
        return base_ptr + ofst;
    }

    /* */

    __host__ __device__ __forceinline__
    T& data_in_cell(int item) {
        return (base_ptr + ofst)[item];
    }

    /*
    __host__ __device__ __forceinline__
    const T& data_in_cell(int item) const {
        return (base_ptr + ofst)[item];
    }
    */

    __host__ __device__ __forceinline__
    T& data_at_cell(int cell, int item) {
        return this[cell].data_in_cell(item);
    }
};

__host__ __device__ __forceinline__
static size_t TAPAS_ALIGN_TO_WARP(size_t S) {
    if (0) {
        return (TAPAS_CEILING((S), 32) * 32);
    } else {
        return S;
    }
}

/* Vector (on a device) consists of concatenation of vectors whose
   contents are in v[0..hunks-1].  (It can have gaps between
   subvectors to align them to the warps). */

template <class T>
struct Vector_Pack {
    using value_type = T;

    Cell_Data<T>* v;
    int hunks;
    int bodies;

    __host__ __device__ __forceinline__
    Vector_Pack(Cell_Data<T>* v, int hunks, int size) {
        this->v = v;
        this->hunks = hunks;
        this->bodies = size;
    }

    __host__ __device__ __forceinline__
    size_t size() {
        return bodies;
    }

    __device__ __forceinline__
    struct Cell_Item_t
    index_in_cell(size_t index) {
        Cell_Item_t r {-1, 0};
        size_t base;
        base = 0;
        for (int c = 0; c < hunks; c++) {
            if (base <= index && index < base + v[c].size) {
                assert(r.cell == -1);
                r.cell = c;
                r.item = (int)(index - base);
                break;
            }
            base += TAPAS_ALIGN_TO_WARP(v[c].size);
        }
        return r;
    }

    __device__ __forceinline__
    bool index_valid(size_t index) {
        struct Cell_Item_t x = index_in_cell(index);
        return (x.cell != -1);
    }

    __device__ __forceinline__
    T& operator[] (size_t index) {
        struct Cell_Item_t x = index_in_cell(index);
        int cell = x.cell;
        int item = x.item;
        if (cell != -1) {
            return v[cell].data_in_cell(item);
        } else {
            return v[0].data_in_cell(0);
        }
    }
};

template <class T>
struct Vector_Raw {
    using value_type = T;

    T* v;
    size_t bodies;

    __host__ __device__ __forceinline__
    Vector_Raw(T* v, size_t size) {
        this->v = v;
        this->bodies = size;
    }

    __host__ __device__ __forceinline__
    size_t size() {
        return bodies;
    }

    __device__ __forceinline__
    T& operator[] (size_t index) {
        return v[index];
    }
};

/**
 * \brief Single argument mapping over bodies on GPU
 */

template <class CA, class V3, class BT, class BT_ATTR, class Fn, class... Args>
__global__
void vectormap_cuda_plain_kernel1(const CA c_attr, const V3 c_center,
                                  const BT* b, BT_ATTR* b_attr,
                                  size_t sz, Fn f, Args... args) {
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  if (index < sz) {
    f(c_attr, c_center, (b + index), (b_attr + index), args...);
  }
}

template <class Fn, class BT, class BT_ATTR, class CELL_ATTR, class VEC,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel1(int nmapdata, size_t nbodies,
                                 CELLDATA<BT>* body_list,
                                 CELLDATA<BT_ATTR>* attr_list,
                                 CELL_ATTR* cell_attrs,
                                 VEC* cell_centers,
                                 Fn f, Args... args) {
    static_assert(std::is_same<BT_ATTR, kvec4>::value, "attribute type=kvec4");

    int index = (blockDim.x * blockIdx.x + threadIdx.x);
    if (index < nbodies) {
        int bodycount = 0;
        int nth = -1;
        for (size_t i = 0; i < nmapdata; i++) {
            if (index < (bodycount + body_list[i].size)) {
                nth = i;
                break;
            } else {
                bodycount += body_list[i].size;
            }
        }
        assert(nth != -1);
        int nthbody = index - bodycount;
        f(cell_attrs[nth], cell_centers[nth],
          (body_list[nth].data + nthbody), (attr_list[nth].data + nthbody),
          args...);
    }
}

/* (Two argument mapping (each pair)) */

/* Accumulates partial acceleration for the 1st vector.  Blocking
   size of the 2nd vector is passed as TILESIZE. */

template <class Fn, class BV, class BA, class... Args>
__global__
void vectormap_cuda_plain_kernel2(BV* v0, BV* v1, BA* a0,
                                  size_t n0, size_t n1, int tilesize,
                                  Fn f, Args... args) {
    assert(tilesize <= blockDim.x);
    int index = (blockDim.x * blockIdx.x + threadIdx.x);
    extern __shared__ BV scratchpad[];
    int ntiles = TAPAS_CEILING(n1, tilesize);
    BV* p0 = ((index < n0) ? &v0[index] : &v0[0]);
    BA q0 = ((index < n0) ? a0[index] : a0[0]);
    for (int t = 0; t < ntiles; t++) {
        if ((tilesize * t + threadIdx.x) < n1 && threadIdx.x < tilesize) {
            scratchpad[threadIdx.x] = v1[tilesize * t + threadIdx.x];
        }
        __syncthreads();

        if (index < n0) {
            unsigned int jlim = min(tilesize, (int)(n1 - tilesize * t));
            /*AHO*/ //#pragma unroll 64
            for (unsigned int j = 0; j < jlim; j++) {
                BV* p1 = &scratchpad[j];
                if (!(v0 == v1 && index == (tilesize * t + j))) {
                    f(p0, p1, q0, args...);
                }
            }
        }
        __syncthreads();
    }
    if (index < n0) {
        a0[index] = q0;
    }
}

template <class Fn, class BODY, class ATTR, class... Args>
__global__
void vectormap_cuda_pack_kernel2(Vector_Pack<BODY> lbody,
                                 Vector_Pack<ATTR> lattr,
                                 int rsize, BODY* rdata, int tilesize,
                                 Fn f, Args... args) {
    assert(tilesize <= blockDim.x);
    int index = (blockDim.x * blockIdx.x + threadIdx.x);
    int ntiles = TAPAS_CEILING(rsize, tilesize);
    extern __shared__ BODY scratchpad[];

    bool valid_index = lbody.index_valid(index);
    BODY& p0 = lbody[index];
    ATTR& a0 = lattr[index];

    ATTR q1 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < ntiles; t++) {
        if ((tilesize * t + threadIdx.x) < rsize && threadIdx.x < tilesize) {
            scratchpad[threadIdx.x] = rdata[tilesize * t + threadIdx.x];
        }
        __syncthreads();

        if (valid_index) {
            unsigned int jlim = min(tilesize, (int)(rsize - tilesize * t));
#pragma unroll 64
            for (unsigned int j = 0; j < jlim; j++) {
                BODY& p1 = scratchpad[j];
                f(p0, a0, const_cast<const BODY&>(p1), const_cast<const ATTR&>(q1), args...);
            }
        }
        __syncthreads();
    }
}

template <int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Base {
    using Body = _BT;
    using BodyAttr = _BT_ATTR;
    using CellAttr = _CELL_ATTR;

    /**
     * \brief Memory allocator for the unified memory.
     * It will replace the
     * vector allocators.  (N.B. Its name should be generic because it
     * is used in CPUs also.)
     */

    template <typename T>
    struct um_allocator : public std::allocator<T> {
    public:
        /*typedef T* pointer;*/
        /*typedef const T* const_pointer;*/
        /*typedef T value_type;*/
        template <class U> struct rebind {typedef um_allocator<U> other;};

        T* allocate(size_t n, const void* hint = 0) {
            T* p;
            p = new T[n];
            //CUDA_SAFE_CALL(cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachGlobal));
            //CUDA_SAFE_CALL(cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachHost));
            assert(p != nullptr);
            //fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd sizeof(T)=%zd size=%zd\n", p, n, sizeof(T), n * sizeof(T)); fflush(0);
            return p;
        }

        void deallocate(T* p, size_t n) {
            CUDA_SAFE_CALL(cudaFree(p));
            //fprintf(stderr, ";; cudaFree() p=%p n=%zd\n", p, n); fflush(0);
        }

        explicit um_allocator() throw() : std::allocator<T>() {}

        /*explicit*/ um_allocator(const um_allocator<T>& a) throw()
            : std::allocator<T>(a) {}

        template <class U> explicit
        um_allocator(const um_allocator<U>& a) throw()
            : std::allocator<T>(a) {}

        ~um_allocator() throw() {}
    };

    /**
     * CUDA GPU device information
     */

    TESLA tesla_device_;
    inline TESLA& tesla_device() { return tesla_device_; }

    /**
     * \brief Setup CUDA devices: allocate 1 GPU per process
     * (considering multiple processes per node)
     */

    void Setup(int cta, int nstreams) {
        assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

        tesla_device_.cta_size = cta;
        tesla_device_.n_streams = nstreams;

#ifdef USE_MPI

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int rankofnode, rankinnode, nprocsinnode;
        rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);
        //printf("rankofnode=%d, rankinnode=%, nprocsinnode=%d\n", rankofnode, rankinnode, nprocsinnode);

#else /* #ifdef USE_MPI */

        int rank = 0;
        int rankinnode = 0;
        int nprocsinnode = 1;

#endif /* USE_MPI */

        SetGPU();

        int ngpus;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&ngpus));
#if 0
        if (ngpus < nprocsinnode) {
            fprintf(stderr, "More ranks than GPUs on a node ngpus = %d, nprocsinnode = %d\n", ngpus, nprocsinnode);
            assert(ngpus >= nprocsinnode);
        }
#endif

        // Since we assume CUDA_VISIBLE_DEVICES is properly set by
        // SetGPU() function or by the user manually, Each process
        // should find 1 GPU.

        assert(ngpus == 1);

        // Fixed. Always use the first GPU (see above).

        tesla_device_.gpuno = 0;
        //tesla_device_.gpuno = rankinnode;
        cudaDeviceProp prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, tesla_device_.gpuno));
        CUDA_SAFE_CALL(cudaSetDevice(tesla_device_.gpuno));

        //printf(";; Rank#%d uses GPU#%d\n", rank, tesla_device_.gpuno); // always GPU#0

        assert(prop.unifiedAddressing);

        tesla_device_.sm = (prop.major * 10 + prop.minor);
        tesla_device_.n_sm = prop.multiProcessorCount;

        tesla_device_.n_cores = 0;
        for (struct TESLA_CORES& i : tesla_cores) {
            if (i.sm == tesla_device_.sm) {
                tesla_device_.n_cores = i.n_cores;
                break;
            }
        }
        assert(tesla_device_.n_cores != 0);

        tesla_device_.scratchpad_size = prop.sharedMemPerBlock;
        tesla_device_.max_cta_size = prop.maxThreadsPerBlock;
        assert(prop.maxThreadsPerMultiProcessor >= prop.maxThreadsPerBlock * 2);

        for (int i = 0; i < tesla_device_.n_streams; i++) {
            CUDA_SAFE_CALL(cudaStreamCreate(&tesla_device_.streams[i]));
        }
    }

    /**
     * \brief Release the CUDA device
     */

    void Release() {
        for (int i = 0; i < tesla_device_.n_streams; i++) {
            CUDA_SAFE_CALL( cudaStreamDestroy(tesla_device_.streams[i]) );
        }
    }
};

template <int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Simple : public Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR> {
    using Body = _BT;
    using BodyAttr = _BT_ATTR;
    using CellAttr = _CELL_ATTR;
    using Base = Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR>;

    /* (One argument mapping) */

    /* NOTE IT RUNS ON CPUs.  The kernel "tapas_kernel::L2P()" is not
       coded to be run on GPUs, since it accesses the cell. */

#if 1
    template <class Fn, class Cell, class... Args>
    void vmap1(Fn f, BodyIterator<Cell> iter, Args... args) {
        //std::cout << "Vectormap_CUDA_Simple::map1() is called. " << iter.size() << std::endl;
        int sz = iter.size();
        for (int i = 0; i < sz; i++) {
            f(*(iter + i), args...);
        }
    }

#else
    // WIP: inactivated.
    template <class Fn, class Cell, class... Args>
    void vmap1(Fn f, BodyIterator<Cell> b0, Args... args) {
        static std::mutex mutex0;
        static struct cudaFuncAttributes tesla_attr0;

        TESLA& tesla = Base::tesla_device();

        if (tesla_attr0.binaryVersion == 0) {
            mutex0.lock();
            CUDA_SAFE_CALL(cudaFuncGetAttributes(
                                                 &tesla_attr0,
                                                 &vectormap_cuda_kernel1<Fn, Body, Args...>));
            mutex0.unlock();
        }
        assert(tesla_attr0.binaryVersion != 0);

        size_t n0 = b0.size();
        int n0up = (TAPAS_CEILING(n0, 256) * 256);
        int ctasize = std::min(n0up, tesla_attr0.maxThreadsPerBlock);
        size_t nblocks = TAPAS_CEILING(n0, ctasize);

        tesla_streamid++;
        int s = (tesla_streamid % tesla.n_streams);
        vectormap_cuda_kernel1<<<nblocks, ctasize, 0, tesla.streams[s]>>>
            (b0, n0, f, args...);
    }
#endif

    /**
     * \brief Two argument mapping
     * Implements a map on a GPU.  It extracts vectors of bodies.  It
     * uses a fixed command stream to serialize processing on each cell.
     * A call to cudaDeviceSynchronize() is needed on the caller of
     * Tapas-map.  The CTA size is the count in the first cell rounded
     * up to multiples of 256.  The tile size is the count in the first
     * cell rounded down to multiples of 64 (tile size is the count of
     * preloading of the second cells).
     */

    template <class Fn, class Cell, class... Args>
    void Plain2(Fn f, Cell& c0, Cell& c1, Args... args) {
        static_assert(std::is_same<Body, typename Cell::BT::type>::value, "inconsistent template arguments");
        static_assert(std::is_same<BodyAttr, typename Cell::BT_ATTR>::value, "inconsistent template arguments");

        using BT = Body;
        using BT_ATTR = BodyAttr;

        auto& data = c0.data();

        // nvcc's bug? the compiler cannot find base class' member function
        // so we need "Base::"
        TESLA& tesla = Base::tesla_device();

        static std::mutex mutex1;
        static struct cudaFuncAttributes tesla_attr1;
        if (tesla_attr1.binaryVersion == 0) {
            mutex1.lock();
            CUDA_SAFE_CALL(cudaFuncGetAttributes(
                                                 &tesla_attr1,
                                                 &vectormap_cuda_plain_kernel2<Fn, BT, BT_ATTR, Args...>));
            mutex1.unlock();
        }
        assert(tesla_attr1.binaryVersion != 0);

        assert(c0.IsLeaf() && c1.IsLeaf());
        /* (Cast to drop const, below). */
        BT* v0 = (BT*)&(c0.body(0));
        BT* v1 = (BT*)&(c1.body(0));
        BT_ATTR* a0 = (BT_ATTR*)&(c0.body_attr(0));
        size_t n0 = c0.nb();
        size_t n1 = c1.nb();
        assert(n0 != 0 && n1 != 0);

        /*bool am = AllowMutual<T1_Iter, T2_Iter>::value(b0, b1);*/
        /*int n0up = (TAPAS_CEILING(n0, 256) * 256);*/
        /*int n0up = (TAPAS_CEILING(n0, 32) * 32);*/
        int cta0 = (TAPAS_CEILING(tesla.cta_size, 32) * 32);
        int ctasize = std::min(cta0, tesla_attr1.maxThreadsPerBlock);
        assert(ctasize == tesla.cta_size);

        int tile0 = (tesla.scratchpad_size / sizeof(Body));
        int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
        int tilesize = std::min(ctasize, tile1);
        assert(tilesize > 0);

        int scratchpadsize = (sizeof(Body) * tilesize);
        size_t nblocks = TAPAS_CEILING(n0, ctasize);

#if 0 /*AHO*/
        fprintf(stderr, "launch arrays=(%p/%ld, %p/%ld, %p/%ld) blks=%ld cta=%d\n",
                v0, n0, v1, n1, a0, n0, nblocks, ctasize);
        fflush(0);
#endif

        // array of bodies for c0
        //BT* bodies0 = c0.IsLocal() ? data.local_bodies_ : data.let_bodies_;

        // array of bodies for c1
        //BT* bodies1 = c1.IsLocal() ? data.local_bodies_ : data.let_bodies_;

        int s = (((unsigned long)&c0 >> 4) % tesla.n_streams);
        vectormap_cuda_plain_kernel2<<<nblocks, ctasize, scratchpadsize,
            tesla.streams[s]>>>
            (v0, v1, a0, n0, n1, tilesize, f, args...);
    }

    /**
     * \fn Vectormap_CUDA_Simple::map2
     * \brief Calls a function FN given by the user on each data pair in the
     *        cells.  f takes arguments of Body&, Body&,
     *        BodyAttr&, and extra call arguments.
     */

    template <class Fn, class Cell, class...Args>
    void vmap2(Fn f, ProductIterator<BodyIterator<Cell>> prod,
               bool mutualinteraction, Args... args) {
        //printf("Vectormap_CUDA_Simple::map2\n"); fflush(0);

        typedef BodyIterator<Cell> Iter;
        const Cell& c0 = prod.first().cell();
        const Cell& c1 = prod.second().cell();
        if (c0 == c1) {
            Plain2(f, c0, c1, args...);
        } else {
            Plain2(f, c0, c1, args...);
            //Plain2(f, c1, c0, args...); // mutual is not supported
        }
    }
};

/**
 * \brief CUDA kernel invoke for 2-parameter Map()
 * Launches a kernel on Tesla.
 * Used by Vectormap_CUDA_Pakced and Applier.
 */

template <class Caller, class Fn, class... Args>
void invoke_kernel2(Caller* caller,
                    Vector_Pack<typename Caller::Body> lbody,
                    Vector_Pack<typename Caller::BodyAttr> lattr,
                    Cell_Data<Body>& r,
                    int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
                    Fn f, Args... args) {
    using BV = typename Caller::Body;
    using BA = typename Caller::BodyAttr;

    TESLA& tesla = caller->tesla_device();
    tesla_streamid++;
    int s = (tesla_streamid % tesla.n_streams);

    vectormap_cuda_pack_kernel2<<<nblocks, ctasize, scratchpadsize,
        tesla.streams[s]>>>
        (lbody, lattr, r.size, r.data(), tilesize, f, args...);
}

// Utility routine to support delayed dispatch of function template
// with variadic parameters.

template <int ...>
struct seq {};

template <int N, int ...S>
struct gens : gens<N-1, N-1, S...> {};

template <int...S>
struct gens<0, S...> {
    typedef seq<S...> type;
};

/**
 * Applier for Map-2 (2-parameter map)
 */

template <class Vectormap>
struct AbstractApplier {
    using Body = typename Vectormap::Body;
    using BodyAttr = typename Vectormap::BodyAttr;
    virtual void invoke(Vectormap* vm,
                        Vector_Pack<Body>& lbody,
                        Vector_Pack<BodyAttr>& lattr,
                        Cell_Data<Body>& r,
                        int tilesize, size_t nblocks, int ctasize,
                        int scratchpadsize) = 0;
    virtual ~AbstractApplier() {}
};

template <class Vectormap, class Fn, class...Args>
struct Kernel2_Thunk : public AbstractApplier<Vectormap> {
    Vectormap* vm;
    Fn f_;
    std::tuple<Args...> args_; // variadic arguments

    using Body = typename Vectormap::Body;
    using BodyAttr = typename Vectormap::BodyAttr;
    //using CellPairs = typename Vectormap::CellPairs;

    Kernel2_Thunk(Vectormap* vm_, Fn f, Args... args)
        : vm (vm_), f_ (f), args_ (args...) {}

#if (ALLPAIRS == 0)

    void invoke(Vectormap* vm,
                Vector_Pack<Body>& lbody,
                Vector_Pack<BodyAttr>& lattr,
                Cell_Data<Body>& r,
                int tilesize, size_t nblocks, int ctasize,
                int scratchpadsize) {
        using ArgsTuple = std::tuple<Args...>;
        invoke_impl(vm, lbody, lattr, r,
                    tilesize, nblocks, ctasize, scratchpadsize,
                    typename gens<std::tuple_size<ArgsTuple>::value>::type {});
    }

    template <int... NL>
    void invoke_impl(Vectormap* vm,
                     Vector_Pack<Body> lbody,
                     Vector_Pack<BodyAttr> lattr,
                     Cell_Data<Body>& r,
                     int tilesize, size_t nblocks, int ctasize,
                     int scratchpadsize,
                     seq<NL...>) {
        ::tapas::invoke_kernel2(vm, lbody, lattr, r,
                                tilesize, nblocks, ctasize, scratchpadsize,
                                f_, std::get<NL>(args_)...);
    }

#else /*ALLPAIRS*/

    void invoke(Vectormap* vm,
                Vector_Pack<Body>& lbody,
                Vector_Pack<BodyAttr>& lattr,
                Cell_Data<Body>& r,
                int tilesize, size_t nblocks, int ctasize,
                int scratchpadsize) {
        TESLA& tesla = vm->tesla_device();
        Vector_Raw<Body> rdata (r.data(), r.size);
        BodyAttr zero = {0.0f, 0.0f, 0.0f, 0.0f};
        int rr[1];
        allpairs(tesla,
                 tilesize, nblocks, ctasize, scratchpadsize,
                 /*h*/ 0, /*g*/ f_, /*z*/ zero,
                 /*r*/ rr, /*x*/ lbody, /*y*/ rdata, /*w*/ lattr);
    }

#endif /*ALLPAIRS*/
};

template <typename T>
struct Mirror_Data {
    T* ddata;
    T* hdata;
    size_t size;

    Mirror_Data() : ddata (nullptr), hdata (nullptr), size (0) {}

    void ensure_size(size_t n) {
        if (size < n) {
            free_data();
            size = n;
            cudaError_t ce;
            ce = cudaMalloc(&this->ddata, (sizeof(T) * n));
            assert(ce == cudaSuccess);
            ce = cudaMallocHost(&this->hdata, (sizeof(T) * n));
            assert(ce == cudaSuccess);
        }
    }

    void free_data() {
        cudaError_t ce;
        ce = cudaFree(this->ddata);
        assert(ce == cudaSuccess);
        ce = cudaFree(this->hdata);
        assert(ce == cudaSuccess);
    }

    void copy_in(size_t n) {
        assert(size == n);
        cudaError_t ce;
        ce = cudaMemcpy(this->ddata, this->hdata, (sizeof(T) * size),
                        cudaMemcpyHostToDevice);
        assert(ce == cudaSuccess);
    }
};

template <int _DIM, typename _FP, typename _BT,
          typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Packed
    : public Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR> {
    using VectorMap = Vectormap_CUDA_Packed<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR>;
    using BT = _BT;
    using BT_ATTR = _BT_ATTR;
    using Body = _BT;
    using Attr = _BT_ATTR;
    using Body_List = Cell_Data<BT>;
    using Attr_List = Cell_Data<BT_ATTR>;
    using Cell_Data_Triple = std::tuple<Body_List, Attr_List, Body_List>;
    using CellPairs = std::vector<Cell_Data_Triple>;

    // Data for 1-parameter Map()

    std::mutex pack1_mutex_;

    // Data for 2-parameter Map()

    std::vector<Cell_Data_Triple> cellpairs2_;
    Mirror_Data<Body_List> body_list2_;
    Mirror_Data<Attr_List> attr_list2_;
    std::mutex pack2_mutex_;

    /* Map arguments (recorded at the first map call.) */

    Body* host_local_bodies_ = 0;
    Body* host_let_bodies_ = 0;
    Attr* host_local_attrs_ = 0;
    Body* dev_local_bodies_ = 0;
    Body* dev_let_bodies_ = 0;
    Attr* dev_local_attrs_ = 0;
    size_t local_nb_ = 0;
    size_t let_nb_ = 0;

    // funct_id_ is used to check if the same Funct is used for all
    // cell pairs.  In the current implementation, it is assumed that
    // a single function and the same optional arguments (= Args...)
    // are used to all cell pairs.

    std::mutex applier2_mutex_;
    intptr_t funct_id_;
    AbstractApplier<VectorMap>* applier2_;

    cudaFuncAttributes func_attrs_;

    double time_device_call_;

    void Start2() {
#ifdef TAPAS_DEBUG
        printf(";; start\n"); fflush(0);
#endif
        cellpairs2_.clear();
        //cellpairs1_.clear();
    }

    /**
     * @brief ctor.
     * not thread safe
     */

    Vectormap_CUDA_Packed()
        : cellpairs2_()
        , body_list2_()
        , attr_list2_()
        , pack1_mutex_()
        , pack2_mutex_()
        , applier2_mutex_()
        , funct_id_(0)
        , applier2_(nullptr)
        , time_device_call_(0)
    {}

    inline std::vector<Cell_Data_Triple>& cellpairs2() {
        return cellpairs2_;
    }

    inline Mirror_Data<Body_List>& body_list2() {
        return body_list2_;
    }

    inline Mirror_Data<Attr_List>& attr_list2() {
        return attr_list2_;
    }

    template <class Fn, class Cell, class... Args>
    inline void vmap1(Fn f, BodyIterator<Cell> iter, Args... args) {
        //std::cout << "Yey! new Vectormap_CUDA_Packed::Map1() is called. " << iter.size() << std::endl;
        int sz = iter.size();
        for (int i = 0; i < sz; i++) {
            f(iter.cell(), *iter, iter.attr(), args...);
            iter++;
        }
    }

    /* (Two argument mapping with left cells packing.) */

    /**
     * \brief Vectormap_CUDA_Packed::map2
     */

    template <class Cell, class Fn, class... Args>
    void vmap2(Fn f, ProductIterator<BodyIterator<Cell>> prod,
               bool mutualinteraction, Args... args) {
        static_assert(std::is_same<typename Cell::Body, Body>::value,
                      "inconsistent Cell and Body types");
        static_assert(std::is_same<typename Cell::BodyAttr, Attr>::value,
                      "inconsistent Cell and BodyAttr types");

        Cell& c0 = prod.first().cell();
        Cell& c1 = prod.second().cell();
        assert(c0.IsLeaf() && c1.IsLeaf());

        if (c0.nb() == 0 || c1.nb() == 0) {
            return;
        }

        // Create Applier with Funct and Args...

        if (applier2_ == nullptr) {
            applier2_mutex_.lock();
            if (applier2_ == nullptr) {
                applier2_ = (new Kernel2_Thunk<VectorMap, Fn, Args...>
                             (this, f, args...));

                funct_id_ = Type2Int<Fn>::value();

                host_local_bodies_ = c0.data().local_bodies_.data();
                host_let_bodies_ = c0.data().let_bodies_.data();
                host_local_attrs_ = c0.data().local_body_attrs_.data();
                dev_local_bodies_ = nullptr;
                dev_let_bodies_ = nullptr;
                dev_local_attrs_ = nullptr;
                local_nb_ = c0.data().local_bodies_.size();
                let_nb_ = c0.data().let_bodies_.size();

                assert(func_attrs_.binaryVersion == 0);
                cudaError_t ce;
                ce = (cudaFuncGetAttributes
                      (&func_attrs_,
                       &vectormap_cuda_pack_kernel2
                       <Fn, Body, Attr, Args...>));
                assert(ce == cudaSuccess);
#if 0
                fprintf(stderr,
                        ";; vectormap_cuda_pack_kernel2:"
                        " binaryVersion=%d, cacheModeCA=%d,"
                        " constSizeBytes=%zd,"
                        " localSizeBytes=%zd, maxThreadsPerBlock=%d,"
                        " numRegs=%d,"
                        " ptxVersion=%d, sharedSizeBytes=%zd\n",
                        func_attrs_.binaryVersion, func_attrs_.cacheModeCA,
                        func_attrs_.constSizeBytes, func_attrs_.localSizeBytes,
                        func_attrs_.maxThreadsPerBlock, func_attrs_.numRegs,
                        func_attrs_.ptxVersion, func_attrs_.sharedSizeBytes);
                fflush(0);
#endif
                assert(func_attrs_.binaryVersion != 0);
            }
            applier2_mutex_.unlock();
        }

        TAPAS_ASSERT(funct_id_ == Type2Int<Fn>::value());

        Cell_Data<BT> d0;
        Cell_Data<BT> d1;
        Cell_Data<BT_ATTR> a0;

        d0.size = c0.nb();
        d0.base_ptr = c0.body_base_ptr();
        d0.ofst = c0.body_offset();

        a0.size = c0.nb();
        a0.base_ptr = c0.body_attr_base_ptr();
        a0.ofst = c0.body_offset();

        d1.size = c1.nb();
        d1.base_ptr = c1.body_base_ptr();
        d1.ofst = c1.body_offset();

        if (c0 == c1) {
            pack2_mutex_.lock();
            cellpairs2_.push_back(Cell_Data_Triple(d0, a0, d1));
            pack2_mutex_.unlock();
        } else {
            pack2_mutex_.lock();
            cellpairs2_.push_back(Cell_Data_Triple(d0, a0, d1));
            // mutual interaction is not supported in this CUDA version.
            //cellpairs2_.push_back(std::tuple<Cell_Data<BT>, Cell_Data<BT_ATTR>, Cell_Data<BT>>(d1, a1, d0));
            pack2_mutex_.unlock();
        }
    }

    /* Limit of the number of threads in grids. */

    static const constexpr int N0 = (16 * 1024);

    void copy_in_h2d() {
        assert(host_local_bodies_ != nullptr
               && host_let_bodies_ != nullptr
               && host_local_attrs_ != nullptr
               && dev_local_bodies_ == nullptr
               && dev_let_bodies_ == nullptr
               && dev_local_attrs_ == nullptr);

        cudaError_t ce;
        ce = cudaMalloc(&dev_local_bodies_, (sizeof(Body) * local_nb_));
        assert(ce == cudaSuccess);
        ce = cudaMalloc(&dev_let_bodies_, (sizeof(Body) * let_nb_));
        assert(ce == cudaSuccess);
        ce = cudaMalloc(&dev_local_attrs_, (sizeof(Attr) * local_nb_));
        assert(ce == cudaSuccess);

        ce = cudaMemcpy(dev_local_bodies_, host_local_bodies_,
                        (sizeof(Body) * local_nb_),
                        cudaMemcpyHostToDevice);
        assert(ce == cudaSuccess);
        ce = cudaMemcpy(dev_let_bodies_, host_let_bodies_,
                        (sizeof(Body) * let_nb_),
                        cudaMemcpyHostToDevice);
        assert(ce == cudaSuccess);
        ce = cudaMemcpy(dev_local_attrs_, host_local_attrs_,
                        (sizeof(Attr) * local_nb_),
                        cudaMemcpyHostToDevice);
        assert(ce == cudaSuccess);

        migrate_cell_data_pointers();
    }

    void migrate_cell_data_pointers() {
        CellPairs& host_data = cellpairs2_;
        for (size_t i = 0; i < host_data.size(); i++) {
            Cell_Data<Body>& trgb = std::get<0>(host_data[i]);
            Cell_Data<Attr>& trga = std::get<1>(host_data[i]);
            Cell_Data<Body>& srcb = std::get<2>(host_data[i]);

            // Rewirte CellData::base_ptr to device memory pointeres.

            // target cell's base_ptr must be SharedData::local_bodies_
            // (because target cells are always local)

            TAPAS_ASSERT(trgb.base_ptr == host_local_bodies_);
            trgb.base_ptr = dev_local_bodies_;

            TAPAS_ASSERT(trga.base_ptr == host_local_attrs_);
            trga.base_ptr = dev_local_attrs_;

            // source cell's base_ptr can be either
            // SharedData::local_bodies_ or let_bodies_

            if (srcb.base_ptr == host_local_bodies_) {
                srcb.base_ptr = dev_local_bodies_;
            } else if (srcb.base_ptr == host_let_bodies_) {
                srcb.base_ptr = dev_let_bodies_;
            } else {
                assert(0);
            }
        }
    }

    void copy_out_d2h() {
        cudaError_t ce;
        ce = cudaMemcpy(host_local_bodies_, dev_local_bodies_,
                        (sizeof(Body) * local_nb_),
                        cudaMemcpyDeviceToHost);
        assert(ce == cudaSuccess);
        ce = cudaMemcpy(host_let_bodies_, dev_let_bodies_,
                        (sizeof(Body) * let_nb_),
                        cudaMemcpyDeviceToHost);
        assert(ce == cudaSuccess);
        ce = cudaMemcpy(host_local_attrs_, dev_local_attrs_,
                        (sizeof(Attr) * local_nb_),
                        cudaMemcpyDeviceToHost);
        assert(ce == cudaSuccess);
        ce = cudaFree(dev_local_bodies_);
        assert(ce == cudaSuccess);
        ce = cudaFree(dev_let_bodies_);
        assert(ce == cudaSuccess);
        ce = cudaFree(dev_local_attrs_);
        assert(ce == cudaSuccess);
        dev_local_bodies_ = nullptr;
        dev_let_bodies_ = nullptr;
        dev_local_attrs_ = nullptr;
    }

    /* Starts launching a kernel on collected cells. */

    void vmap_on_collected2() {
        logger::startTimer("Direct part");

        double t1 = MPI_Wtime();

        TAPAS_ASSERT(applier2_ != nullptr);
        auto vm = this;
        assert(vm->cellpairs2_.size() != 0);
#ifdef TAPAS_DEBUG
        printf(";; pairs=%ld\n", vm->cellpairs2_.size());
#endif

        TESLA& tesla = vm->tesla_device();
        int cta0 = (TAPAS_CEILING(tesla.cta_size, 32) * 32);
        int ctasize = std::min(cta0, func_attrs_.maxThreadsPerBlock);
        assert(ctasize == tesla.cta_size);
        int tile0 = (tesla.scratchpad_size / sizeof(Body));
        int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
        int tilesize = std::min(ctasize, tile1);
        assert(tilesize > 0);
        int scratchpadsize = (sizeof(Body) * tilesize);
        size_t nblocks = TAPAS_CEILING(N0, ctasize);

        copy_in_h2d();

        auto cellpairs = vm->cellpairs2_;

        size_t nn = cellpairs.size();
        vm->body_list2().ensure_size(nn);
        vm->attr_list2().ensure_size(nn);

        auto t3 = std::chrono::high_resolution_clock::now();

        auto comp = cellcompare_r<Cell_Data<Body>, Cell_Data<Attr>, Cell_Data<Body>>();
        std::sort(cellpairs.begin(), cellpairs.end(), comp);

        for (size_t i = 0; i < nn; i++) {
            Cell_Data_Triple& c = cellpairs[i];
            vm->body_list2().hdata[i] = std::get<0>(c);
            vm->attr_list2().hdata[i] = std::get<1>(c);
        }

        vm->body_list2().copy_in(nn);
        vm->attr_list2().copy_in(nn);

        auto t4 = std::chrono::high_resolution_clock::now();

        Cell_Data<Body> rcell = std::get<2>(cellpairs[0]);
        int cells = 0;
        int bodies = 0;

        for (size_t i = 0; i < nn; i++) {
            Cell_Data_Triple& c = cellpairs[i];
            Cell_Data<Body>& r = std::get<2>(c);
            if (rcell.data() != r.data()) {
                assert(i != 0 && cells > 0);
                int s = (i - cells);
                Vector_Pack<Body> lbody (&body_list2_.ddata[s], cells, bodies);
                Vector_Pack<Attr> lattr (&attr_list2_.ddata[s], cells, bodies);
                applier2_->invoke(vm, lbody, lattr, rcell,
                                  tilesize, nblocks, ctasize, scratchpadsize);
                cells = 0;
                bodies = 0;
                rcell = r;
            }
            Cell_Data<Body>& l = std::get<0>(c);
            size_t nb = TAPAS_CEILING((bodies + l.size), ctasize);
            if (nb > nblocks) {
                assert(i != 0 && cells > 0);
                int s = (i - cells);
                Vector_Pack<Body> lbody (&body_list2_.ddata[s], cells, bodies);
                Vector_Pack<Attr> lattr (&attr_list2_.ddata[s], cells, bodies);
                applier2_->invoke(vm, lbody, lattr, rcell,
                                  tilesize, nblocks, ctasize, scratchpadsize);
                cells = 0;
                bodies = 0;
                rcell = r;
            }
            cells++;
            bodies += TAPAS_ALIGN_TO_WARP(l.size);
        }
        assert(cells > 0);
        int s = (nn - cells);
        Vector_Pack<Body> lbody (&body_list2_.ddata[s], cells, bodies);
        Vector_Pack<Attr> lattr (&attr_list2_.ddata[s], cells, bodies);
        applier2_->invoke(vm, lbody, lattr, rcell,
                          tilesize, nblocks, ctasize, scratchpadsize);

        double time_mcopy = (std::chrono::duration<double>(t4 - t3)).count();

        copy_out_d2h();

        vectormap_check_error("Vectormap_CUDA_Packed::end", __FILE__, __LINE__);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double t2 = MPI_Wtime();

        time_device_call_ = t2 - t1;
        logger::stopTimer("Direct part");
    }

    void Finish2() {
#ifdef TAPAS_DEBUG
        printf(";; Vectormap_CUDA_Packed::Finish2\n"); fflush(0);
#endif

        vmap_on_collected2();

        if (applier2_ != nullptr) {
            applier2_mutex_.lock();
            if (applier2_ != nullptr) {
                delete applier2_;
                applier2_ = nullptr;
                funct_id_ = 0;
            }
            applier2_mutex_.unlock();
        }
    }
};

BR1_
#undef BR0_
#undef BR1_

#endif /*__CUDACC__*/

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
