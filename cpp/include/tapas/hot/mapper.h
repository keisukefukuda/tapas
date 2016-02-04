#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include <chrono>

#include "tapas/iterator.h"
#include "tapas/hot/let.h"

namespace tapas {
namespace hot {

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using clock = std::chrono::system_clock;

/**
 * @brief Helper subroutine called from Mapper::Map
 */ 
template<class Mapper, class T1_Iter, class T2_Iter, class Funct, class...Args>
static void ProductMapImpl(Mapper &mapper,
                           T1_Iter iter1, int beg1, int end1,
                           T2_Iter iter2, int beg2, int end2,
                           Funct f, Args... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);
  
  using CellType = typename T1_Iter::CellType;
  using Th = typename CellType::Threading;

  const constexpr int kT1 = T1_Iter::kThreadSpawnThreshold;
  const constexpr int kT2 = T2_Iter::kThreadSpawnThreshold;

  bool am = iter1.AllowMutualInteraction(iter2);
  
  if (end1 - beg1 <= kT1 || end2 - beg2 <= kT2) {
    // The two ranges (beg1,end1) and (beg2,end2) are fine enough to apply f in a serial manner.

    // Create a function object to be given to the Container's Map function.
    //typedef std::function<void(C1&, C2&)> Callback;
    //Callback gn = [&args...](C1 &c1, C2 &c2) { f()(c1, c2, args...); };
    for(int i = beg1; i < end1; i++) {
      for(int j = beg2; j < end2; j++) {
        T1_Iter lhs = iter1 + i;
        T2_Iter rhs = iter2 + j;
        // if i and j are mutually interactive, f(i,j) is evaluated only once.

        //bool am = lhs.AllowMutualInteraction(rhs);
        
        if ((am && i <= j) || !am) {
          if (lhs.IsLocal()) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
            mapper.Map(f, lhs, rhs, args...);
          }
        }
      }
    }
  } else {
    int mid1 = (end1 + beg1) / 2;
    int mid2 = (end2 + beg2) / 2;
    // run (beg1,mid1) x (beg2,mid2) and (mid1,end1) x (mid2,end2) in parallel
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, mid2, f, args...); });
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, mid1, end1, iter2, mid2, end2, f, args...); });
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() {ProductMapImpl(mapper, iter1, beg1, mid1, iter2, mid2, end2, f, args...);});
      tg.createTask([&]() {ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, mid2, f, args...);});
      tg.wait();
    }
  }
}

template<class Cell, class Body, class LET>
struct CPUMapper {
  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }

  /**
   *
   */
  template <class Funct, class... Args>
  void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    using Th = typename Cell::Threading;
    typename Th::TaskGroup tg;

    TAPAS_ASSERT(iter.index() == 0);
    
    for (int i = 0; i < iter.size(); i++) {
      tg.createTask([=]() mutable { this->Map(f, *iter, args...); });
      iter++;
    } 
    tg.wait();
  }

  // cell x cell
  template <class Funct, class...Args>
  void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    using Time = decltype(clock::now());
    Time net_bt, net_et;
    
    // All Map() function traverse starts from (root, root) for LET construction and GPU init/finalize
    if (c1.IsRoot() && c2.IsRoot()) {
      auto bt = clock::now();
      
      if (c1.data().mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
        char t[] = "TAPAS_IN_LET=1";
        putenv(t); // to avoid warning "convertion from const char* to char*"
#endif
        LET::Exchange(c1, f, args...);
#ifdef TAPAS_DEBUG
        unsetenv("TAPAS_IN_LET");
#endif
      }
      auto et = clock::now();
      c1.data().time_map2_let = duration_cast<milliseconds>(et - bt).count() * 1e-3;
      
      net_bt = clock::now();
    }

    // Actual Map() operation
    f(c1, c2, args...);
    
    if (c1.IsRoot() && c2.IsRoot()) {
      net_et = clock::now();
      c1.data().time_map2_net = duration_cast<milliseconds>(net_et - net_bt).count() * 1e-3;
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }
  
  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // bodies
  template <class Funct, class... Args>
  void Map(Funct f, BodyIterator<Cell> iter, Args...args) {
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
  }
  
  // body x body 
  template<class Funct, class...Args>
  void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(*b1, b1.attr(), *b2, b2.attr(), args...);
  }

  inline void Setup() {  }
  
  inline void Start() {  }

  inline void Finish() {  }
}; // class CPUMapper


#ifdef __CUDACC__

#include "tapas/vectormap.h"
#include "tapas/vectormap_cuda.h"

template<class Cell, class Body, class LET>
struct GPUMapper {

  using Data = typename Cell::Data;
  using Vectormap = tapas::Vectormap_CUDA_Packed<Cell::Dim, typename Cell::FP, typename Cell::Body, typename Cell::BodyAttr>;

  Vectormap vmap_;

  std::chrono::high_resolution_clock::time_point map2_all_beg_, map2_all_end_;

  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  /* (Specialization of the Map() below by a general ProductIterator<T>
     with ProductIterator<BodyIterator<T>>). */
  template <class Funct, class...Args>
  inline void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    vmap_.map2(f, prod, args...);
  }

  inline void Setup() {
    vmap_.setup(64,31);
  }

  // GPUMapper::Start for 2-param Map()
  inline void Start() {
    vmap_.start();
  }

  // GPUMapper::Finish for 2-param Map()
  inline void Finish() {
    vmap_.finish();
  }

  /**
   *
   */
  template <class Funct, class... Args>
  inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    using Th = typename Cell::Threading;
    typename Th::TaskGroup tg;

    TAPAS_ASSERT(iter.index() == 0);
    
    for (int i = 0; i < iter.size(); i++) {
      tg.createTask([=]() mutable { this->Map(f, *iter, args...); });
      iter++;
    } 
    tg.wait();
  }

  /**
   * @brief Initialization of 2-param Map()
   *
   * - Setup CUDA device and variables
   * - Construct & exchange LET
   */ 
  template <class Funct, class...Args>
  void Map2Init(Funct f, Cell&c1, Cell&c2, Args...args) {
    auto &data = c1.data();
    map2_all_beg_ = std::chrono::high_resolution_clock::now();

    // -- Perform LET exchange if more than 1 MPI process
    if (c1.data().mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
      char t[] = "TAPAS_IN_LET=1";
      putenv(t); // to avoid warning "convertion from const char* to char*"
#endif
      LET::Exchange(c1, f, args...);
        
#ifdef TAPAS_DEBUG
      unsetenv("TAPAS_IN_LET");
#endif
    } else {
      // mpi_size_ == 0
      data.time_let_all = data.time_let_traverse
                        = data.time_let_req
                        = data.time_let_response
                        = data.time_let_register
                        = 0;
    }

    // -- initialize GPU
    Start();

    // -- check
    if (c1.GetOptMutual()) {
      std::cerr << "[To Fix] Error: mutual is not supported in CUDA implementation" << std::endl;
      //exit(-1);
    }
    data.time_map2_let = data.time_let_all;
  }

  /**
   * @brief Finalization of 2-param Map()
   *
   * - Execute CUDA kernel on the interaction list
   * - Collect time information
   */ 
  template <class Funct, class...Args>
  void Map2Finish(Funct, Cell &c1, Cell &c2, Args...) {
    auto &data = c1.data();
    Finish(); // Execute CUDA kernel

    // collect runtime information
    map2_all_end_  = std::chrono::high_resolution_clock::now();
    auto d = map2_all_end_ - map2_all_beg_;
    
    data.time_map2_dev = vmap_.time_device_call_;
    data.time_map2_all = std::chrono::duration_cast<std::chrono::microseconds>(d).count() * 1e-6;
  }
  
  // GPUMapper::Map
  // cell x cell
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    static std::chrono::high_resolution_clock::time_point t1, t2;
    
    if (c1.IsRoot() && c2.IsRoot()) {
      Map2Init(f, c1, c2, args...);
    }
    
    f(c1, c2, args...);
    
    if (c1.IsRoot() && c2.IsRoot()) {
      Map2Finish(f, c1, c2, args...);
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }
  
  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // bodies
  template <class Funct, class... Args>
  inline void Map(Funct f, BodyIterator<Cell> iter, Args...args) {
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
  }
  
  // body x body 
  template<class Funct, class...Args>
  inline void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(*b1, b1.attr(), *b2, b2.attr(), args...);
  }

}; // class GPUMapper

#endif /* __CUDACC__ */

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_

