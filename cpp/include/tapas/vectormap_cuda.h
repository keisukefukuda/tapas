/* vectormap_cuda.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_CUDA_H_
#define TAPAS_VECTORMAP_CUDA_H_

/** @file vectormap_cuda.h @brief Direct part by CUDA.  See
    "vectormap_cpu.h". */

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#include "tapas/vectormap_util.h"

#include <atomic>
#include <mutex>

namespace tapas {

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

static struct TESLA {
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
} tesla_dev;

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

static void vectormap_check_error(const char *msg, const char *file, const int line) {
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

/* (Single argument mapping) */

template <class CA, class V3, class BT, class BT_ATTR, class Funct, class... Args>
__global__
void vectormap_cuda_kernel1(const CA c_attr, const V3 c_center,
                            const BT* b, BT_ATTR* b_attr,
                            size_t sz, Funct f, Args... args) {
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  if (index < sz) {
    f(c_attr, c_center, (b + index), (b_attr + index), args...);
  }
}

template <class Funct, class BT, class BT_ATTR, class CELL_ATTR, class VEC,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel1(int nmapdata, size_t nbodies,
                                 CELLDATA<BT>* body_list,
                                 CELLDATA<BT_ATTR>* attr_list,
                                 CELL_ATTR* cell_attrs,
                                 VEC* cell_centers,
                                 Funct f, Args... args) {
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

template <class Funct, class BT, class BT_ATTR, class... Args>
__global__
void vectormap_cuda_plain_kernel2(BT* v0, BT* v1, BT_ATTR* a0,
                                  size_t n0, size_t n1, int tilesize,
                                  Funct f, Args... args) {
  assert(tilesize <= blockDim.x);
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  extern __shared__ BT scratchpad[];
  int ntiles = TAPAS_CEILING(n1, tilesize);
  BT* p0 = ((index < n0) ? &v0[index] : &v0[0]);
  BT_ATTR q0 = ((index < n0) ? a0[index] : a0[0]);
  for (int t = 0; t < ntiles; t++) {
    if ((tilesize * t + threadIdx.x) < n1 && threadIdx.x < tilesize) {
      scratchpad[threadIdx.x] = v1[tilesize * t + threadIdx.x];
    }
    __syncthreads();

    if (index < n0) {
      unsigned int jlim = min(tilesize, (int)(n1 - tilesize * t));
#pragma unroll 128
      for (unsigned int j = 0; j < jlim; j++) {
        BT* p1 = &scratchpad[j];
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

/* (Atomic-add code from cuda-c-programming-guide). */

__device__
static double atomicAdd(double* address, double val) {
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

__device__
static double atomicAdd(float* address, float val) {
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

template <class Funct, class BT, class BT_ATTR,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel2(CELLDATA<BT>* v, CELLDATA<BT_ATTR>* a,
                                 size_t nc,
                                 int rsize, BT* rdata, int tilesize,
                                 Funct f, Args... args) {
  static_assert(std::is_same<BT_ATTR, kvec4>::value, "attribute type=kvec4");

  assert(tilesize <= blockDim.x);
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  extern __shared__ BT scratchpad[];

  int cell = -1;
  int item = 0;
  int base = 0;
  for (int c = 0; c < nc; c++) {
    if (base <= index && index < base + v[c].size) {
      assert(cell == -1);
      cell = c;
      item = (index - base);
    }
    base += (TAPAS_CEILING(v[c].size, 32) * 32);
  }

  int ntiles = TAPAS_CEILING(rsize, tilesize);
  BT &p0 = (cell != -1) ? v[cell].data[item] : v[0].data[0];
  BT_ATTR q0 = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int t = 0; t < ntiles; t++) {
    if ((tilesize * t + threadIdx.x) < rsize && threadIdx.x < tilesize) {
      scratchpad[threadIdx.x] = rdata[tilesize * t + threadIdx.x];
    }
    __syncthreads();

    if (cell != -1) {
      unsigned int jlim = min(tilesize, (int)(rsize - tilesize * t));
#pragma unroll 128
      for (unsigned int j = 0; j < jlim; j++) {
        BT &p1 = scratchpad[j];
        f(&p0, &p1, q0, args...);
      }
    }
    __syncthreads();
  }

  if (cell != -1) {
    assert(item < a[cell].size);
    BT_ATTR &a0 = a[cell].data[item];
    atomicAdd(&(a0[0]), q0[0]);
    atomicAdd(&(a0[1]), q0[1]);
    atomicAdd(&(a0[2]), q0[2]);
    atomicAdd(&(a0[3]), q0[3]);
  }
}

template<class T0, class T1, class T2>
struct cellcompare_r {
  bool operator() (const std::tuple<T0, T1, T2> &i,
                   const std::tuple<T0, T1, T2> &j) {
    return ((std::get<2>(i).data) < (std::get<2>(j).data));
  }
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Base {
  typedef typename _BI::type BT;
  typedef _BT_ATTR BT_ATTR;
  typedef _CELL_ATTR CELL_ATTR;

  /** Memory allocator for the unified memory.  It will replace the
      vector allocators.  (N.B. Its name should be generic because it
      is used in CPUs also.) */

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
      fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd\n", p, n); fflush(0);
      return p;
    }

    void deallocate(T* p, size_t n) {
      cudaError_t ce = cudaFree(p);
      assert(ce == cudaSuccess);
      fprintf(stderr, ";; cudaFree() p=%p n=%zd\n", p, n); fflush(0);
    }

    explicit um_allocator() throw() : std::allocator<T>() {}

    /*explicit*/ um_allocator(const um_allocator<T> &a) throw()
      : std::allocator<T>(a) {}

    template <class U> explicit
    um_allocator(const um_allocator<U> &a) throw()
      : std::allocator<T>(a) {}

    ~um_allocator() throw() {}
  };

  static void vectormap_setup(int cta, int nstreams) {
    assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

    tesla_dev.cta_size = cta;
    tesla_dev.n_streams = nstreams;

    /*AHO*/ /* USE PROPER WAY TO KNOW OF USE OF MPI. */

#ifdef EXAFMM_TAPAS_MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rankofnode, rankinnode, nprocsinnode;
    rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);

#else /*EXAFMM_TAPAS_MPI*/

    int rank = 0;
    int rankinnode = 0;
    int nprocsinnode = 1;

#endif /*EXAFMM_TAPAS_MPI*/

    cudaError_t ce;
    int ngpus;
    ce = cudaGetDeviceCount(&ngpus);
    assert(ce == cudaSuccess);
    if (ngpus < nprocsinnode) {
      fprintf(stderr, "More ranks than GPUs on a node\n");
      assert(ngpus >= nprocsinnode);
    }

    tesla_dev.gpuno = rankinnode;
    cudaDeviceProp prop;
    ce = cudaGetDeviceProperties(&prop, tesla_dev.gpuno);
    assert(ce == cudaSuccess);
    ce = cudaSetDevice(tesla_dev.gpuno);
    assert(ce == cudaSuccess);

    printf(";; Rank#%d uses GPU#%d\n", rank, tesla_dev.gpuno);

    assert(prop.unifiedAddressing);

    tesla_dev.sm = (prop.major * 10 + prop.minor);
    tesla_dev.n_sm = prop.multiProcessorCount;

    tesla_dev.n_cores = 0;
    for (struct TESLA_CORES &i : tesla_cores) {
      if (i.sm == tesla_dev.sm) {
        tesla_dev.n_cores = i.n_cores;
        break;
      }
    }
    assert(tesla_dev.n_cores != 0);

    tesla_dev.scratchpad_size = prop.sharedMemPerBlock;
    tesla_dev.max_cta_size = prop.maxThreadsPerBlock;
    assert(prop.maxThreadsPerMultiProcessor >= prop.maxThreadsPerBlock * 2);

    for (int i = 0; i < tesla_dev.n_streams; i++) {
      ce = cudaStreamCreate(&tesla_dev.streams[i]);
      assert(ce == cudaSuccess);
    }
  }

  static void vectormap_release() {
    for (int i = 0; i < tesla_dev.n_streams; i++) {
      cudaError_t ce = cudaStreamDestroy(tesla_dev.streams[i]);
      assert(ce == cudaSuccess);
    }
  }

  static void vectormap_start() {}

  static void vectormap_end() {
    vectormap_check_error("vectormap_end", __FILE__, __LINE__);
    cudaError_t ce;
    ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
      fprintf(stderr,
              "%s:%i (%s): CUDA ERROR (%d): %s.\n",
              __FILE__, __LINE__, "cudaDeviceSynchronize",
              (int)ce, cudaGetErrorString(ce));
      assert(ce == cudaSuccess);
    }
  }

  template <class Funct, class... Args>
  static void vectormap_finish(Funct f, Args... args) {}
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Simple_Map1
  : Vectormap_CUDA_Base<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR> {
  typedef typename _BI::type BT;
  typedef _BT_ATTR BT_ATTR;
  typedef _CELL_ATTR CELL_ATTR;

  /* (One argument mapping) */

  template <class Funct, class Cell, class... Args>
  static void vector_map1(Funct f, BodyIterator<Cell> b0,
                          Args... args) {
    typedef typename Cell::TSPClass TSP;

    static std::mutex mutex0;
    static struct cudaFuncAttributes tesla_attr0;
    if (tesla_attr0.binaryVersion == 0) {
      mutex0.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr0,
        &vectormap_cuda_kernel1<typename Cell::ATTR,
        tapas::Vec<TSP::Dim, typename TSP::FP>,
        typename Cell::BodyType, typename Cell::BodyAttrType,
        Funct, Args...>);
      assert(ce == cudaSuccess);
      mutex0.unlock();
    }
    assert(tesla_attr0.binaryVersion != 0);

    int sz = b0.size();
    const Cell c = (*b0).cell();
    const typename Cell::ATTR c_attr = c.attr();
    const tapas::Vec<TSP::Dim, typename TSP::FP> c_center = c.center();
    const typename Cell::BodyType* b = b0.as_body();
    typename Cell::BodyAttrType* b_attr = &b0.attr();
    if (0) {
      /*AHO*/ /* (Run on CPU). */
      for (int i = 0; i < sz; i++) {
        f(c_attr, c_center, (b + i), (b_attr + i), args...);
      }
    } else {
      int szup = (TAPAS_CEILING(sz, 256) * 256);
      int ctasize = std::min(szup, tesla_attr0.maxThreadsPerBlock);
      size_t nblocks = TAPAS_CEILING(sz, ctasize);

      /*AHO*/
      if (0) {
        fprintf(stderr, "launch array=(%p/%d, %p/%d) blks=%ld cta=%d\n",
                b, sz, b_attr, sz, nblocks, ctasize);
        fflush(0);
      }

      streamid++;
      int s = (streamid % tesla_dev.n_streams);
      vectormap_cuda_kernel1<<<nblocks, ctasize, 0, tesla_dev.streams[s]>>>
        (c_attr, c_center, b, b_attr, sz, f, args...);
    }
  }
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Simple_Map2
  : Vectormap_CUDA_Base<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR> {
  typedef typename _BI::type BT;
  typedef _BT_ATTR BT_ATTR;
  typedef _CELL_ATTR CELL_ATTR;

  /* (Two argument mapping) */

  /* Implements a map on a GPU.  It extracts vectors of bodies.  It
     uses a fixed command stream to serialize processing on each cell.
     A call to cudaDeviceSynchronize() is needed on the caller of
     Tapas-map.  The CTA size is the count in the first cell rounded
     up to multiples of 256.  The tile size is the count in the first
     cell rounded down to multiples of 64 (tile size is the count of
     preloading of the second cells). */

  template <class Funct, class Cell, class... Args>
  static void vectormap_cuda_plain(Funct f, Cell &c0, Cell &c1,
                                   Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value),
                  "inconsistent template arguments");

    static std::mutex mutex1;
    static struct cudaFuncAttributes tesla_attr1;
    if (tesla_attr1.binaryVersion == 0) {
      mutex1.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr1,
        &vectormap_cuda_plain_kernel2<Funct, BT, BT_ATTR, Args...>);
      assert(ce == cudaSuccess);
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
    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr1.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    int tile0 = (tesla_dev.scratchpad_size / sizeof(typename Cell::BodyType));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(typename Cell::BodyType) * tilesize);
    size_t nblocks = TAPAS_CEILING(n0, ctasize);

    /*AHO*/
    if (0) {
      fprintf(stderr, "launch arrays=(%p/%ld, %p/%ld, %p/%ld)"
              " blks=%ld cta=%d\n",
              v0, n0, v1, n1, a0, n0, nblocks, ctasize);
      fflush(0);
    }

    int s = (((unsigned long)&c0 >> 4) % tesla_dev.n_streams);
    vectormap_cuda_plain_kernel2<<<nblocks, ctasize, scratchpadsize,
      tesla_dev.streams[s]>>>
      (v0, v1, a0, n0, n1, tilesize, f, args...);
  }

  /** Calls a function FN given by the user on each data pair in the
      cells.  FN takes arguments of Cell::BodyType&, Cell::BodyType&,
      Cell::BodyAttrType&, and extra call arguments. */

  template <class Funct, class Cell, class...Args>
  static void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                          Args... args) {
    printf("vector_map2X\n"); fflush(0);

    typedef BodyIterator<Cell> Iter;
    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    if (c0 == c1) {
      vectormap_cuda_plain(f, c0, c1, args...);
    } else {
      vectormap_cuda_plain(f, c0, c1, args...);
      vectormap_cuda_plain(f, c1, c0, args...);
    }
  }
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Simple
: Vectormap_CUDA_Simple_Map1<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR>,
  Vectormap_CUDA_Simple_Map2<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR>
{};

/* Vector of data (mainly bodies and body attrs) copied between the
   host and the device. */

template <class T>
struct Cell_Data {
  int size;
  T* data;
};

/* Allocated data for copying between the host and the device; One for
   the host and the other for the device. */

template <class T>
struct Mirror_Data {
  T* ddata;
  T* hdata;
  size_t size;

  void assure_size(size_t n) {
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

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Packed_Map1
  : Vectormap_CUDA_Base<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR> {
  typedef typename _BI::type BT;
  typedef _BT_ATTR BT_ATTR;
  typedef _CELL_ATTR CELL_ATTR;

  typedef tapas::Vec<_DIM, _FP> VEC;
  typedef Cell_Data<BT> Body_List;
  typedef Cell_Data<BT_ATTR> Attr_List;
  typedef std::tuple<Body_List, Attr_List, CELL_ATTR, VEC> MapData1;

  /* Limit of the number of threads in grids. */

  static const int N0 = (16 * 1024);

  /* (Single argument mapping.) */

  /* STATIC MEMBER FIELDS. (It is a trick.  See:
     http://stackoverflow.com/questions/11709859/) */

  static std::mutex &pack1_mutex() {
    static std::mutex pack1_mutex_;
    return pack1_mutex_;
  }

  static std::vector<MapData1> &mapdata1() {
    static std::vector<MapData1> mapdata1_;
    return mapdata1_;
  }

  static Mirror_Data<Body_List> &body_lists() {
    static Mirror_Data<Body_List> body_lists_;
    return body_lists_;
  }

  static Mirror_Data<Attr_List> &attr_lists() {
    static Mirror_Data<Attr_List> attr_lists_;
    return attr_lists_;
  }

  static Mirror_Data<_CELL_ATTR> &cell_attrs() {
    static Mirror_Data<_CELL_ATTR> cell_attrs_;
    return cell_attrs_;
  }

  static Mirror_Data<VEC> &cell_centers() {
    static Mirror_Data<VEC> cell_centers_;
    return cell_centers_;
  }

  /* (Single argument mapping.) */

  template <class Funct, class Cell, class... Args>
  static void vector_map1(Funct f, BodyIterator<Cell> b0,
                          Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value
                   && std::is_same<_CELL_ATTR, typename Cell::ATTR>::value),
                  "inconsistent template arguments");

    const Cell c = (*b0).cell();
    if (c.nb() == 0) {return;}

    /* (Cast to drop const, below). */

    const typename Cell::ATTR c_attr = c.attr();
    const VEC c_center = c.center();
    /*const typename Cell::BodyType* b = b0.as_body();*/
    /*typename Cell::BodyAttrType* b_attr = &b0.attr();*/
    Cell_Data<BT> d0;
    Cell_Data<BT_ATTR> a0;
    size_t sz = c.nb();
    d0.data = (BT*)&(c.body(0));
    d0.size = sz;
    a0.data = (BT_ATTR*)&(c.body_attr(0));
    a0.size = sz;

    pack1_mutex().lock();
    mapdata1().push_back(MapData1(d0, a0, c_attr, c_center));
    pack1_mutex().unlock();
  }

  /* Launches a kernel on Tesla. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_invoke1(int nmapdata, size_t nbodies,
                                size_t nblocks, int ctasize,
                                int scratchpadsize,
                                Cell &dummy, Funct f, Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value
                   && std::is_same<_CELL_ATTR, typename Cell::ATTR>::value),
                  "inconsistent template arguments");

    /*AHO*/
    if (0) {
      printf("kernel(nblocks=%ld ctasize=%d scratchpadsize=%d tilesize=%d\n",
             nblocks, ctasize, scratchpadsize, 0);
      printf("invoke1(ncells=%d, nbodies=%ld)\n", nmapdata, nbodies);
      for (int i = 0; i < nmapdata; i++) {
        Cell_Data<BT> &c = std::get<0>(mapdata1()[i]);
        printf("celll=%p[%d]\n", c.data, c.size);
      }
      fflush(0);
    }

    streamid++;
    int s = (streamid % tesla_dev.n_streams);
    vectormap_cuda_pack_kernel1<<<nblocks, ctasize, scratchpadsize,
      tesla_dev.streams[s]>>>
      (nmapdata, nbodies,
       body_lists().ddata, attr_lists().ddata,
       cell_attrs().ddata, cell_centers().ddata, f, args...);
  }

  /* Starts launching a kernel on collected cells. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_on_collected1(Funct f, Cell dummy, Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value
                   && std::is_same<_CELL_ATTR, typename Cell::ATTR>::value),
                  "inconsistent template arguments");

    assert(mapdata1().size() != 0);

    static std::mutex mutex2;
    static struct cudaFuncAttributes tesla_attr2;
    if (tesla_attr2.binaryVersion == 0) {
      mutex2.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr2,
        &vectormap_cuda_pack_kernel1<Funct, BT, BT_ATTR, CELL_ATTR, VEC,
        Cell_Data, Args...>);
      assert(ce == cudaSuccess);
      mutex2.unlock();
      if (0) {
        /*GOMI*/
        printf((";; vectormap_cuda_pack_kernel1:"
                " binaryVersion=%d, cacheModeCA=%d, constSizeBytes=%zd,"
                " localSizeBytes=%zd, maxThreadsPerBlock=%d, numRegs=%d,"
                " ptxVersion=%d, sharedSizeBytes=%zd\n"),
               tesla_attr2.binaryVersion, tesla_attr2.cacheModeCA,
               tesla_attr2.constSizeBytes, tesla_attr2.localSizeBytes,
               tesla_attr2.maxThreadsPerBlock, tesla_attr2.numRegs,
               tesla_attr2.ptxVersion, tesla_attr2.sharedSizeBytes);
        fflush(0);
      }
    }
    assert(tesla_attr2.binaryVersion != 0);

    //printf(";; pairs=%ld\n", mapdata1().size());

    size_t nn = mapdata1().size();
    size_t nbodies = 0;
    for (size_t i = 0; i < nn; i++) {
      MapData1 &c = mapdata1()[i];
      nbodies += std::get<0>(c).size;
    }

    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr2.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    size_t nblocks = TAPAS_CEILING(nbodies, ctasize);

    body_lists().assure_size(nn);
    attr_lists().assure_size(nn);
    cell_attrs().assure_size(nn);
    cell_centers().assure_size(nn);

    for (size_t i = 0; i < nn; i++) {
      MapData1 &c = mapdata1()[i];
      body_lists().hdata[i] = std::get<0>(c);
      attr_lists().hdata[i] = std::get<1>(c);
      cell_attrs().hdata[i] = std::get<2>(c);
      cell_centers().hdata[i] = std::get<3>(c);
    }
    body_lists().copy_in(nn);
    attr_lists().copy_in(nn);
    cell_attrs().copy_in(nn);
    cell_centers().copy_in(nn);

    vectormap_invoke1(nn, nbodies, nblocks, ctasize, 0,
                      dummy, f, args...);
  }
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Packed_Map2
  : Vectormap_CUDA_Base<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR> {
  typedef typename _BI::type BT;
  typedef _BT_ATTR BT_ATTR;

  typedef Cell_Data<BT> Body_List;
  typedef Cell_Data<BT_ATTR> Attr_List;
  typedef std::tuple<Body_List, Attr_List, Body_List> MapData2;

  /* Limit of the number of threads in grids. */

  static const int N0 = (16 * 1024);

  /* STATIC MEMBER FIELDS. (It is a trick.  See:
     http://stackoverflow.com/questions/11709859/) */

  static std::mutex &pack2_mutex() {
    static std::mutex pack2_mutex_;
    return pack2_mutex_;
  }

  static std::vector<MapData2> &cellpairs() {
    static std::vector<MapData2> cellpairs_;
    return cellpairs_;
  }

  static Mirror_Data<Body_List> &body_lists2() {
    static Mirror_Data<Body_List> body_lists2_;
    return body_lists2_;
  }

  static Mirror_Data<Attr_List> &attr_lists2() {
    static Mirror_Data<Attr_List> attr_lists2_;
    return attr_lists2_;
  }

  /* (Two argument mapping with left packing.) */

  template <class Funct, class Cell, class... Args>
  static void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                          Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value),
                  "inconsistent template arguments");

    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    assert(c0.IsLeaf() && c1.IsLeaf());

    if (c0.nb() == 0 || c1.nb() == 0) return;

    /* (Cast to drop const, below). */

    Cell_Data<BT> d0;
    Cell_Data<BT> d1;
    Cell_Data<BT_ATTR> a0;
    Cell_Data<BT_ATTR> a1;
    d0.size = c0.nb();
    d0.data = (BT*)&(c0.body(0));
    a0.size = c0.nb();
    a0.data = (BT_ATTR*)&(c0.body_attr(0));
    d1.size = c1.nb();
    d1.data = (BT*)&(c1.body(0));
    a1.size = c1.nb();
    a1.data = (BT_ATTR*)&(c1.body_attr(0));

    if (c0 == c1) {
      pack2_mutex().lock();
      cellpairs().push_back(MapData2(d0, a0, d1));
      pack2_mutex().unlock();
    } else {
      pack2_mutex().lock();
      cellpairs().push_back(MapData2(d0, a0, d1));
      cellpairs().push_back(MapData2(d1, a1, d0));
      pack2_mutex().unlock();
    }
  }

  /* Launches a kernel on Tesla. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_invoke2(int start, int nc, Cell_Data<BT> &r,
                                int tilesize,
                                size_t nblocks, int ctasize,
                                int scratchpadsize,
                                Cell &dummy, Funct f, Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value),
                  "inconsistent template arguments");

    /*AHO*/
    if (0) {
      printf("kernel(nblocks=%ld ctasize=%d scratchpadsize=%d tilesize=%d\n",
             nblocks, ctasize, scratchpadsize, tilesize);
      printf("invoke(start=%d ncells=%d)\n", start, nc);
      for (int i = 0; i < nc; i++) {
        Cell_Data<BT> &lc = std::get<0>(cellpairs()[start + i]);
        Cell_Data<BT_ATTR> &ac = std::get<1>(cellpairs()[start + i]);
        Cell_Data<BT> &rc = std::get<2>(cellpairs()[start + i]);
        assert(rc.data == r.data);
        assert(ac.size == lc.size);
        printf("pair(celll=%p[%d] cellr=%p[%d])\n",
               lc.data, lc.size, rc.data, rc.size);
      }
      fflush(0);
    }

    streamid++;
    int s = (streamid % tesla_dev.n_streams);
    vectormap_cuda_pack_kernel2<<<nblocks, ctasize, scratchpadsize,
      tesla_dev.streams[s]>>>
      (&(body_lists2().ddata[start]), &(attr_lists2().ddata[start]),
       nc, r.size, r.data,
       tilesize, f, args...);
  }

  /* Starts launching a kernel on collected cells. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_on_collected2(Funct f, Cell dummy, Args... args) {
    static_assert((std::is_same<BT, typename Cell::BT::type>::value
                   && std::is_same<BT_ATTR, typename Cell::BT_ATTR>::value),
                  "inconsistent template arguments");

    assert(cellpairs().size() != 0);

    static std::mutex mutex3;
    static struct cudaFuncAttributes tesla_attr3;
    if (tesla_attr3.binaryVersion == 0) {
      mutex3.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr3,
        &vectormap_cuda_pack_kernel2<Funct, BT, BT_ATTR, Cell_Data, Args...>);
      assert(ce == cudaSuccess);
      mutex3.unlock();
      if (0) {
        /*GOMI*/
        printf((";; vectormap_cuda_pack_kernel2:"
                " binaryVersion=%d, cacheModeCA=%d, constSizeBytes=%zd,"
                " localSizeBytes=%zd, maxThreadsPerBlock=%d, numRegs=%d,"
                " ptxVersion=%d, sharedSizeBytes=%zd\n"),
               tesla_attr3.binaryVersion, tesla_attr3.cacheModeCA,
               tesla_attr3.constSizeBytes, tesla_attr3.localSizeBytes,
               tesla_attr3.maxThreadsPerBlock, tesla_attr3.numRegs,
               tesla_attr3.ptxVersion, tesla_attr3.sharedSizeBytes);
        fflush(0);
      }
    }
    assert(tesla_attr3.binaryVersion != 0);

    //printf(";; pairs=%ld\n", cellpairs().size());

    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr3.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    int tile0 = (tesla_dev.scratchpad_size / sizeof(typename Cell::BodyType));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(typename Cell::BodyType) * tilesize);
    size_t nblocks = TAPAS_CEILING(N0, ctasize);

    size_t nn = cellpairs().size();
    body_lists2().assure_size(nn);
    attr_lists2().assure_size(nn);

    std::sort(cellpairs().begin(), cellpairs().end(),
              cellcompare_r<Cell_Data<BT>, Cell_Data<BT_ATTR>, Cell_Data<BT>>());
    for (size_t i = 0; i < nn; i++) {
      MapData2 &c = cellpairs()[i];
      body_lists2().hdata[i] = std::get<0>(c);
      attr_lists2().hdata[i] = std::get<1>(c);
    }
    body_lists2().copy_in(nn);
    attr_lists2().copy_in(nn);

    Cell_Data<BT> xr = std::get<2>(cellpairs()[0]);
    int xncells = 0;
    int xndata = 0;
    for (size_t i = 0; i < nn; i++) {
      MapData2 &c = cellpairs()[i];
      Cell_Data<BT> &r = std::get<2>(c);
      if (xr.data != r.data) {
        assert(i != 0 && xncells > 0);
        vectormap_invoke2((i - xncells), xncells, xr,
                          tilesize, nblocks, ctasize, scratchpadsize,
                          dummy, f, args...);
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      Cell_Data<BT> &l = std::get<0>(c);
      size_t nb = TAPAS_CEILING((xndata + l.size), ctasize);
      if (nb > nblocks) {
        assert(i != 0 && xncells > 0);
        vectormap_invoke2((i - xncells), xncells, xr,
                          tilesize, nblocks, ctasize, scratchpadsize,
                          dummy, f, args...);
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      xncells++;
      xndata += (TAPAS_CEILING(l.size, 32) * 32);
    }
    assert(xncells > 0);
    vectormap_invoke2((nn - xncells), xncells, xr,
                      tilesize, nblocks, ctasize, scratchpadsize,
                      dummy, f, args...);
  }
};

template<int _DIM, class _FP, class _BI, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CUDA_Packed
#if 1
  : Vectormap_CUDA_Packed_Map1<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR>,
#else
  : Vectormap_CUDA_Simple_Map1<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR>,
#endif
  Vectormap_CUDA_Packed_Map2<_DIM, _FP, _BI, _BT_ATTR, _CELL_ATTR> {
  typedef Vectormap_CUDA_Packed_Map1<_DIM, _FP, _BI, _BT_ATTR,
                                     _CELL_ATTR> Map1;
  typedef Vectormap_CUDA_Packed_Map2<_DIM, _FP, _BI, _BT_ATTR,
                                     _CELL_ATTR> Map2;

  static void vectormap_start() {
    //printf(";; vectormap_start\n"); fflush(0);
    Map1::mapdata1().clear();
    Map2::cellpairs().clear();
  }

  template <class Funct, class Cell, class... Args>
  static void vectormap_finish1(Funct f, Cell dummy, Args... args) {
    //printf(";; vectormap_finish1\n"); fflush(0);
    if (Map1::mapdata1().size() != 0) {
      Map1::vectormap_on_collected1(f, dummy, args...);
    }
  }

  template <class Funct, class Cell, class... Args>
  static void vectormap_finish2(Funct f, Cell dummy, Args... args) {
    //printf(";; vectormap_finish2\n"); fflush(0);
    if (Map2::cellpairs().size() != 0) {
      Map2::vectormap_on_collected2(f, dummy, args...);
    }
  }
};

}

#endif /*__CUDACC__*/

#endif /*TAPAS_VECTORMAP_CUDA_H_*/
