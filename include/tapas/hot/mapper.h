#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include <iostream>
#include <cxxabi.h>

#include "tapas/util.h"
#include "tapas/iterator.h"

extern "C" {
  // for performance debugging
  void myth_start_papi_counter(const char*);
  void myth_stop_papi_counter(void);
}


#if !defined(__CUDACC__) // not CUDA
/**
 * Check if user's function f is mutual or non-mutual.
 * These classes are inactivated in CUDA mode since nvcc's C++11 support is not
 * sufficient and this code does not compile.
 */
template<class Funct, class Cell>
struct CheckMutualCell {
  using ft = tapas::util::function_traits<decltype(&Funct::template operator()<Cell>)>;
  using param1 = typename std::tuple_element<0, typename ft::arity>::type;
  using param2 = typename std::tuple_element<1, typename ft::arity>::type;

  // If the second parameter is const reference, it's non-mutual
  static const constexpr bool value = std::is_same<param1, param2>::value && !std::is_same<param2, const Cell&>::value;
};

template<class Funct, class Body, class BodyAttr>
struct CheckMutualBody {
  using ft = tapas::util::function_traits<decltype(&Funct::template operator()<Body, BodyAttr>)>;
  using param1 = typename std::tuple_element<0, typename ft::arity>::type;
  using param3 = typename std::tuple_element<2, typename ft::arity>::type;

  // If the second parameter is const reference, it's non-mutual
  static const constexpr bool value = std::is_same<param1, param3>::value && !std::is_same<param3, const Body&>::value;
};

#else /* defined(__CUDACC__) */

template<class Funct, class Cell>
struct CheckMutualCell {
  static const constexpr bool value = false;
};

template<class Funct, class Body, class BodyAttr>
struct CheckMutualBody {
  static const constexpr bool value = false;
};
#endif

namespace tapas {
namespace hot {

template<class Cell, class Body, class LET, class Insp1> struct CPUMapper;

/**
 * @brief Helper subroutine called from Mapper::Map, the body of Map-2 operation
 */
template<class Mapper, class T1_Iter, class T2_Iter, class Funct, class...Args>
static void ProductMapImpl(Mapper &mapper,
                           T1_Iter iter1, int beg1, int end1,
                           T2_Iter iter2, int beg2, int end2,
                           Funct f, Args&&... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);

  using CellType = typename T1_Iter::CellType;
  using Th = typename CellType::Threading;

  const constexpr int kT1 = T1_Iter::kThreadSpawnThreshold;
  const constexpr int kT2 = T2_Iter::kThreadSpawnThreshold;

  // If the user's function f is mutual or not.
  // (Determined from the constness of its parameters)
  constexpr bool kMutual = CheckMutualCell<Funct, CellType>::value
                      && std::is_same<typename T1_Iter::value_type,
                                      typename T2_Iter::value_type>::value;
#if 0
  // Debug code
  std::string T1_str, T2_str;
  {
    int status;
    char * t1_demangled = abi::__cxa_demangle(typeid(T1_Iter).name(),0,0,&status);
    char * t2_demangled = abi::__cxa_demangle(typeid(T2_Iter).name(),0,0,&status);
    if (strncmp("tapas::iterator::BodyIterator", t1_demangled, strlen("tapas::iterator::BodyIterator")) != 0 ||
        strncmp("tapas::iterator::BodyIterator", t2_demangled, strlen("tapas::iterator::BodyIterator")) != 0) {
      std::cout << "T1_Iter=" << (t1_demangled+17) << " "
                << "T2_Iter=" << (t2_demangled+17) << " "
                << "iter1.size()=" << iter1.size() << "[" << beg1 << "-" << end1 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << "iter2.size()=" << iter2.size() << "[" << beg2 << "-" << end2 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << ((end1 - beg1 <= kT1 && end2 - beg2 <= kT2) ? "Serial" : "Split")
                << std::endl;
    }
    T1_str = t1_demangled;
    T2_str = t2_demangled;
    free(t1_demangled);
    free(t2_demangled);
  }
#endif

  if (!iter1.SpawnTask()
      || (end1 - beg1 == 1)
      || (end1 - beg1 <= kT1 && end2 - beg2 <= kT2)) {
    // Not to spawn tasks, run in serial
    // The two ranges (beg1,end1) and (beg2,end2) are fine enough to apply f in a serial manner.

    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j < end2; j++) {
        T1_Iter lhs = iter1 + i;
        T2_Iter rhs = iter2 + j;
        // if i and j are mutually interactive, f(i,j) is evaluated only once.

        bool mtl = (kMutual && iter1.cell() == iter2.cell());
                  
        if ((mtl && i <= j) || !mtl) {
          if (lhs.IsLocal()) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
            mapper.Map(f, lhs, rhs, std::forward<Args>(args)...);
          }
        }
      }
    }
  } else if (!kMutual && end2 - beg2 == 1) {
    // NOTE: end1 - beg1 > 1.
    // Source side (iter2) can be split and paralleilzed.
    // target side cannot paralleize due to accumulation
    int mid1 = (end1 + beg1) / 2;

    typename Th::TaskGroup tg;
    tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, end2, f, std::forward<Args>(args)...); });
    ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, end2, f, std::forward<Args>(args)...);
    tg.wait();
  } else if (kMutual && end2 - beg2 == 1) {
    // mutual == 1 && end2 - beg2 == 1
    int mid1 = (end1 + beg1) / 2;
    ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, end2, f, std::forward<Args>(args)...);
    ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, end2, f, std::forward<Args>(args)...);
  } else {
    int mid1 = (end1 + beg1) / 2;
    int mid2 = (end2 + beg2) / 2;
    // run (beg1,mid1) x (beg2,mid2) and (mid1,end1) x (mid2,end2) in parallel
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, mid2, f, std::forward<Args>(args)...); });
      ProductMapImpl(mapper, iter1, mid1, end1, iter2, mid2, end2, f, std::forward<Args>(args)...);
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, mid2, end2, f, std::forward<Args>(args)...); });
      ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, mid2, f, std::forward<Args>(args)...);
      tg.wait();
    }
  }
}

/**
 * \brief Overloaded version of ProductMapImpl for bodies x bodies.
 */
template<class CELL, class BODY, class LET, class INSP1, class Funct, class...Args>
static void ProductMapImpl(CPUMapper<CELL, BODY, LET, INSP1> & /*mapper*/,
                           typename CELL::BodyIterator iter1,
                           int beg1, int end1,
                           typename CELL::BodyIterator iter2,
                           int beg2, int end2,
                           Funct f, Args&&... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);
  //using BodyIterator = typename CELL::BodyIterator;

  using Body = typename CELL::Body;
  using BodyAttr = typename CELL::BodyAttr;
  bool mutual = CheckMutualBody<Funct, Body, BodyAttr>::value && (iter1.cell() == iter2.cell());

  CELL &c1 = iter1.cell();
  CELL &c2 = iter2.cell();

  c1.WeightLf((end1 - beg1) * (end2 - beg2));
  if (mutual) {
    c2.WeightLf((end1 - beg1) * (end2 - beg2));
  }
  
  //auto data = c1.data_ptr();
  auto *bodies1 = &c1.body(0);
  auto *bodies2 = &c2.body(0);
  auto *attrs1 = &c1.body_attr(0);
  auto *attrs2 = &c2.body_attr(0);

  if (mutual) {
    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j <= i; j++) {
        if (1) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
          f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
        }
      }
    }
  } else {
    // non-mutual
    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j < end2; j++) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
        f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
      }
    }
  }
}

template<class Cell, class Body, class LET, class INSP1>
struct CPUMapper {
  enum class Map1Dir {
    None,
    Upward,
    Downward
  };

  Map1Dir map1_dir_;

  std::string label_;
  
  using KeyType = typename Cell::KeyType;
  using SFC = typename Cell::SFC;
  using Data = typename Cell::Data;
  using Insp1 = INSP1;

  CPUMapper() : map1_dir_(Map1Dir::None), label_() { }

  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void MapP2(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args&&...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, std::forward<Args>(args)...);
    }
  }

  template <class Funct, class T1_Iter, class ...Args>
  inline void MapP1(Funct f, ProductIterator<T1_Iter> prod, Args&&...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, std::forward<Args>(args)...);
    }
  }

  // before running upward traversal from the root,
  // we need to run local upward first and communicate the global leaf values between processes.
  template<class Funct, class...Args>
  inline void LocalUpwardMap(Funct f, Cell &c, Args&&...args) {
    auto &data = c.data();

    MPI_Barrier(MPI_COMM_WORLD); // debug

    // Apply the algorithm function f to local trees, of which roots are in data.lroots_
    for (auto &k : data.lroots_) {
      // TODO: parallelizable?
      TAPAS_ASSERT(data.ht_.count(k) == 1);
      Cell &lrc = *data.ht_[k]; // local root cell

      if (!lrc.IsLeaf()) {
        auto iter = lrc.subcells();
        for (index_t i = 0; i < iter.size(); i++) {
          // TODO: parallelization
          //KeyType ck = SFC::Child(lrc.key(), i);
          f(lrc, *iter, std::forward<Args>(args)...);
          iter++;
        }
      } else { // lrc.IsLeaf()
        // Upward algorithm function takes two cells: parent and child.
        // The problem here is that if a local root is just a leaf,
        // the cell is not yet processed. 
        // Thus, such parent/child pairs must be computed before the communication.
        assert(lrc.IsRoot() == false);
        Cell &p = lrc.parent();
        f(p, lrc, std::forward<Args>(args)...);
        data.local_upw_results_[p.key()] = p.attr();
      }
    }
    
    Cell::ExchangeGlobalLeafAttrs(data.ht_gtree_, data.lroots_);
  }

  template<class...Args>
  std::string Concat(Args...args) {
    std::stringstream ss;
    return Concat(ss, args...);
  }
  
  template<class T, class...Args>
  std::string Concat(std::stringstream &ss, T v, Args... args) {
    ss << v;
    return Concat(ss, args...);
  }

  std::string Concat(std::stringstream &ss) {
    return ss.str();
  }

  std::string Concat() {
    return "";
  }
  
  template<class Funct, class...Args>
  void GetFuncLabel(Funct &f, Args&&...args) {
    using G = tapas::util::GetLabel<Funct>;
    
    if (G::HasLabel()) {
      label_ = G::Value(f);
    } else {
      label_ = Concat(std::forward<Args>(args)...);
    }
  }

  /**
   * \brief Upward for root cells
   */
  template<class Funct, class...Args>
  inline void UpwardRoot(Funct f, tapas::iterator::SubCellIterator<Cell> &iter, Args&&...args) {
    Cell &c = iter.cell();
    auto &data = c.data();
    map1_dir_ = Map1Dir::Upward;
    data.local_upw_results_.clear();

    // Local Upward
    if (data.mpi_size_ > 1) {
      double bt = MPI_Wtime();

      LocalUpwardMap(f, c, std::forward<Args>(args)...); // Run local upward first
            
      double et = MPI_Wtime();
      data.time_rec_.Record(data.timestep_, label_ + "-local", et - bt);
    }

    // Global upward
    double bt = MPI_Wtime();
    for (index_t i = 0; i < iter.size(); i++) {
      // TODO: parallelization
      f(c, *iter, std::forward<Args>(args)...);
      iter++;
    }
    double et = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, label_ + "-global", et - bt);
    
    data.local_upw_results_.clear();
    map1_dir_ = Map1Dir::None; // Upward is done.
  }

  /**
   * \brief Upward for non-root cells
   */
  template<class Funct, class ...Args>
  inline void UpwardNonRoot(Funct f, tapas::iterator::SubCellIterator<Cell> &iter, Args&&...args) {
    Cell &c = iter.cell();
    auto &data = c.data();
    
    // Write back the results computed in LocalUpwardMap().
    if (data.local_upw_results_.count(c.key()) != 0) {
      c.attr() = data.local_upw_results_[c.key()];
      data.local_upw_results_.erase(c.key());
    }

    // Traversal for local trees is already done in StartUpwardMap().
    // We just perform traversal in the global tree part.
    // TODO: parallelization
    if (data.gleaves_.count(c.key()) == 0) { // Do not call f and stop traversal on global leaves.
      for (index_t i = 0; i < iter.size(); i++) {
        Cell &parent = c;
        Cell &child = *iter;
        
        // Skip such parent/child pairs (the child is  a global leaf as well as a real leaf)
        // because already processes in LocalUpwardMap(). See LocalUpwardMap() for details.
        if (!(data.gleaves_.count(child.key()) != 0 && child.IsLeaf())) {
          f(parent, child, std::forward<Args>(args)...);
        }
        iter++;
      }
    }
  }
  

  template<class Funct, class...Args>
  inline void DownwardRoot(Funct f, tapas::iterator::SubCellIterator<Cell> &iter, Args&&...args) {
    Cell &c = iter.cell();
    auto &data = c.data();
    
    map1_dir_ = Map1Dir::Downward;

    double bt = MPI_Wtime();
    for (index_t i = 0; i < iter.size(); i++) {
      // TODO: parallelization
      f(c, *iter, std::forward<Args>(args)...);
      iter++;
    }
    double et = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, label_ + "-global", et - bt);
    
    map1_dir_ = Map1Dir::None;
  }

  template<class Funct, class...Args>
  inline void DownwardNonRoot(Funct f, tapas::iterator::SubCellIterator<Cell> &iter, Args&&...args) {
    Cell &c = iter.cell();
    auto &data = c.data();
    
    // non-local cells are eliminated in Map(SubcellIterator).
    for (index_t i = 0; i < iter.size(); i++) {
      // TODO: parallelization
      KeyType ck = SFC::Child(c.key(), i);
      if (data.ht_.count(ck) > 0) {
        f(c, *iter, std::forward<Args>(args)...);
      }
      iter++;
    }
  }

  /**
   * CPUMapper::Map (Bodies)
   * Apply f to `all` bodies in parallel
   */
  template<class Funct, class...Args>
  inline void Map(Funct f, tapas::iterator::Bodies<Cell> bodies, Args&&...args) {
    Data &data = bodies.cell().data();
    size_t nb = data.local_bodies_.size();

    for (size_t i = 0; i < nb; i++) {
      f(data.local_bodies_[i], data.local_body_attrs_[i], std::forward<Args>(args)...);
    }
  }

  /**
   * CPUMapper::Map  (1-parameter)
   * Map-1 with SubCelliterator is for Upward or Downward operation.
   * First, determine the direction (up/down) and if up start from local upward.
   */
  template<class Funct, class...Args>
  inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args&&...args) {
    Cell &c = iter.cell();
    auto &data = c.data();

    if (c.IsRoot()) {
      if (c.IsRoot() && c.IsLeaf()) {
        // which means there is only a single cell in the region (i.e. ncrit > #bodies)
        // does nothing.
        return;
      }

      // Find the label of f.
      // (for example, "DTT", "Upward", etc.
      GetFuncLabel(f, "Map2-", data.count_map2_);
      
      // Map() has just started.
      // Find the direction of the function f (upward or downward)
      if (map1_dir_ != Map1Dir::None) {
        std::cerr << "Tapas ERROR: Tapas' internal state seems to be corrupted. Map function is not thread-safe." << std::endl;
        abort();
      }
      
      double find_bt = MPI_Wtime();
      auto dir = Insp1::FindMap1Direction(c, f, std::forward<Args>(args)...);
      double find_et = MPI_Wtime();
      data.time_rec_.Record(data.timestep_, "Map1-finddir", find_et - find_bt);

      // note for MAP1_UNKNOWN:
      // The Inspector checks if the function make modification on the parent or child.
      // If f does not write to any of them, we assume that it only writes leaves or
      // non-destructive function (like debug printer).
      
      switch(dir) {
        case Insp1::MAP1_UP:      UpwardRoot<Funct, Args...>(f, iter, std::forward<Args>(args)...); break;
        case Insp1::MAP1_DOWN:    DownwardRoot<Funct, Args...>(f, iter, std::forward<Args>(args)...); break;
        case Insp1::MAP1_UNKNOWN: DownwardRoot<Funct, Args...>(f, iter, std::forward<Args>(args)...); break;
        default:
          // This should not happen.
          assert(0);
          break;
      }
      // Count how many times Map-1 is called
      data.count_map1_++;
      
    } else { // for non-root cells
      // Non-root cells
      switch(map1_dir_) {
        case Map1Dir::Upward:   UpwardNonRoot<Funct, Args...>(f, iter, std::forward<Args>(args)...); break;
        case Map1Dir::Downward: DownwardNonRoot<Funct, Args...>(f, iter, std::forward<Args>(args)...); break;
        default:
          std::cerr << "Tapas ERROR: Tapas' internal state seems to be corrupted. Map function is not thread-safe." << std::endl;
          abort();
      }
    }
  }
  

  // cell x cell
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args&&... args) {
    SCOREP_USER_REGION_DEFINE(trav_handle)
        double exec_bt = 0, exec_et = 0; // executor's begin time, end time

    auto &data = c1.data();

    //c2.data().trav_used_src_key_.insert(c2.key());

    if (c1.IsRoot() && c2.IsRoot()) {
      // Pre-traverse procedure
      // All Map() function traverse starts from (root, root)
      // for LET construction and GPU init/finalize

      GetFuncLabel(f, "Map1-", data.count_map2_);

      double insp_bt = MPI_Wtime();
      
      if (data.mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
        char t[] = "TAPAS_IN_LET=1"; // to avoid warning "convertion from const char* to char*"
        putenv(t); 
#endif

        // LET Inspector routine
        LET::Exchange(c1, f, std::forward<Args>(args)...);
        
#ifdef TAPAS_DEBUG
        unsetenv("TAPAS_IN_LET");
#endif
      }

      SCOREP_USER_REGION_BEGIN(trav_handle, "NetTraverse", SCOREP_USER_REGION_TYPE_COMMON);

      double insp_et = MPI_Wtime();
      data.time_rec_.Record(data.timestep_, label_ + "-insp", insp_et - insp_bt);

      exec_bt = MPI_Wtime();
    }

    // myth_start_papi_counter("PAPI_FP_OPS");
    // Body of traverse
#if 1
    f(c1, c2, std::forward<Args>(args)...);
#else
    # warning "Ignoreing user function"
#endif
    // myth_stop_papi_counter();

    if (c1.IsRoot() && c2.IsRoot()) {
      // Post-traverse procedure
      exec_et = MPI_Wtime();
      c1.data().time_rec_.Record(data.timestep_, label_ + "-exec", exec_et - exec_bt);
      SCOREP_USER_REGION_END(trav_handle);
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, c1, *c2, std::forward<Args>(args)...);
  }

  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, c1, *c2, std::forward<Args>(args)...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  template<class Funct, class...Args>
  inline void Map(Funct f, Cell& c, Args&&...args) {
    //std::cout << "Map(F, Cell) is called" << std::endl;
    f(c, std::forward<Args>(args)...);
  }

  // bodies
  template <class Funct, class...Args>
  void Map(Funct f, BodyIterator<Cell> iter, Args&&...args) {
    for (size_t i = 0; i < iter.size(); ++i) {
      f(iter.cell(), *iter, iter.attr(), std::forward<Args>(args)...);
      iter++;
    }
  }

  // body x body
  template<class Funct, class...Args>
  void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args&&...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(*b1, b1.attr(), *b2, b2.attr(), std::forward<Args>(args)...);
  }

  inline void Setup() {  }

  inline void Start2() { }

  inline void Finish() {  }

}; // class CPUMapper


#ifdef __CUDACC__

#include "tapas/vectormap.h"
#include "tapas/vectormap_cuda.h"

template<class Cell, class Body, class LET, class Insp1>
struct GPUMapper : CPUMapper<Cell, Body, LET, Insp1> {

  using Base = CPUMapper<Cell, Body, LET, Insp1>;
  using Data = typename Cell::Data;
  using Vectormap = tapas::Vectormap_CUDA_Packed<Cell::Dim,
                                                 typename Cell::FP,
                                                 typename Cell::Body,
                                                 typename Cell::BodyAttr,
                                                 typename Cell::CellAttr>;

  Vectormap vmap_;

  double map2_all_beg_;
  double map2_all_end_;

  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void MapP2(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args&&...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, std::forward<Args>(args)...);
    }
  }

  template<class Funct, class...Args>
  inline void MapP2(Funct f, ProductIterator<BodyIterator<Cell>, BodyIterator<Cell>> prod, Args&&...args) {
    //std::cout << "MapP2 (body)" << std::endl;

    if (prod.size() > 0) {
      vmap_.map2(f, prod, std::forward<Args>(args)...);
    }
  }

  template <class Funct, class T1_Iter, class ...Args>
  inline void MapP1(Funct f, ProductIterator<T1_Iter> prod, Args&&...args) {
    if (prod.size() > 0) {
      //vmap_.map2(f, prod, args...);
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, std::forward<Args>(args)...);
    }
  }

  template <class Funct, class ...Args>
  inline void MapP1(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args&&...args) {
    if (prod.size() > 0) {
      vmap_.map2(f, prod, std::forward<Args>(args)...);
    }
  }

  GPUMapper() : CPUMapper<Cell, Body, LET, Insp1>() { }

  /**
   * \brief Specialization of Map() over body x body product for GPU
   */
  template <class Funct, class...Args>
  inline void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args&&...args) {
    // Offload bodies x bodies interaction to GPU
    vmap_.map2(f, prod, std::forward<Args>(args)...);
  }

  inline void Setup() {
    // Called in tapas/hot.h
    vmap_.Setup(64,31);
  }

  // GPUMapper::Start for 2-param Map()
  inline void Start2() {
    //std::cout << "*** Start2()" << std::endl;
    vmap_.Start2();
  }

  // GPUMapper::Finish for 2-param Map()
  inline void Finish() {
    //std::cout << "*** Finish()" << std::endl;
    vmap_.Finish2();
  }

  /**
   * @brief Initialization of 2-param Map()
   *
   * - Setup CUDA device and variables
   * - Construct & exchange LET
   */
  template <class Funct, class...Args>
  void Map2_Init(Funct f, Cell&c1, Cell&c2, Args&&...args) {
    auto &data = c1.data();
    map2_all_beg_ = MPI_Wtime();

    // -- Perform LET exchange if more than 1 MPI process
    if (c1.data().mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
      char t[] = "TAPAS_IN_LET=1";
      putenv(t); // to avoid warning "convertion from const char* to char*"
#endif
      LET::Exchange(c1, f, std::forward<Args>(args)...);

#ifdef TAPAS_DEBUG
      unsetenv("TAPAS_IN_LET");
#endif
    }

    // -- initialize GPU for 2-Param Map()
#ifdef TAPAS_DEBUG
    std::cout << "Calling Start2()" << std::endl;
#endif
    Start2();
    
    // -- check
  }

  /**
   * @brief Finalization of 2-param Map()
   *
   * - Execute CUDA kernel on the interaction list
   * - Collect time information
   */
  template <class Funct, class...Args>
  void Map2_Finish(Funct, Cell &c1, Cell &c2, Args&&...) {
    auto &data = c1.data();
    Finish(); // Execute CUDA kernel

    data.time_rec_.Record(data.timestep_, "Map2-device", vmap_.time_device_call_);

    // collect runtime information
    map2_all_end_  = MPI_Wtime();
    auto dt = map2_all_end_ - map2_all_beg_;
    data.time_rec_.Record(data.timestep_, "Map2-all", dt);
  }
  
  /*
   * \brief Main routine of dual tree traversal (2-param Map())
   * GPUMapper::Map
   *
   * Cell x Cell
   */
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args&&... args) {
    static double t1, t2

    //std::cout << "GPUMapper::Map(2)  " << c1.key() << ", " << c2.key() << std::endl;

    if (c1.IsRoot() && c2.IsRoot()) {
      Map2_Init(f, c1, c2, std::forward<Args>(args)...);
    }

    f(c1, c2, std::forward<Args>(args)...);

    if (c1.IsRoot() && c2.IsRoot()) {
      Map2_Finish(f, c1, c2, std::forward<Args>(args)...);
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, c1, *c2, std::forward<Args>(args)...);
  }

  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, c1, *c2, std::forward<Args>(args)...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args&&...args) {
    Map(f, *c1, *c2, std::forward<Args>(args)...);
  }

  // bodies
  template <class Funct, class... Args>
  inline void Map(Funct f, BodyIterator<Cell> iter, Args&&...args) {
#if 0
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), std::forward<Args>(args)...);
      iter++;
    }
#else
    vmap_.map1(f, iter, std::forward<Args>(args)...);
#endif
  }

  template<class Funct, class...Args>
  inline void Map(Funct f, Cell &c, Args&&...args) {
    Base::Map(f, c, std::forward<Args>(args)...);
  }

  /**
   * GPUMapper::Map (Bodies)
   * Delegate to CPUMapper::Map
   */
  template<class Funct, class...Args>
  inline void Map(Funct f, tapas::iterator::Bodies<Cell> bodies, Args&&...args) {
    Base::Map(f, bodies, std::forward<Args>(args)...);
  }
  
  /**
   * GPUMapper::Map (subcelliterator)
   */
  template <class Funct, class... Args>
  inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args&&...args) {
    // nvcc cannot find this function in the Base (=CPUMapper) class, so it's explicitly written
    Base::Map(f, iter, std::forward<Args>(args)...);
  }

  template<class Funct, class...Args>
  inline void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args&&...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    std::cout << "*** Map(body) Wrong one!!" << std::endl;
    abort();
    f(*b1, b1.attr(), *b2, b2.attr(), std::forward<Args>(args)...);
  }
}; // class GPUMapper

#endif /* __CUDACC__ */

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_
