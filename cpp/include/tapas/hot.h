/**
 * @file hot.h
 * @brief Implements MPI-based, SFC (Space filling curves)-based HOT (Hashed Octree) implementation of Tapas
 */
#ifndef TAPAS_HOT_
#define TAPAS_HOT_

#include "tapas/stdcbug.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator> // for std::back_inserter
#include <limits>
#include <list>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>
#include <mpi.h>

#include "tapas/bitarith.h"
#include "tapas/cell.h"
#include "tapas/debug_util.h"
#include "tapas/geometry.h"
#include "tapas/iterator.h"
#include "tapas/logging.h"
#include "tapas/mpi_util.h"
#include "tapas/sfc_morton.h"
#include "tapas/threading/default.h"

#include "tapas/hot/buildtree.h"
#include "tapas/hot/global_tree.h"
#include "tapas/hot/insp1.h"
#include "tapas/hot/mapper.h"
#include "tapas/hot/report.h"
#include "tapas/hot/shared_data.h"
#include "tapas/hot/cell.h"
#include "tapas/hot/partitioner.h"

#ifdef __CUDACC__
#else
# include "tapas/vectormap.h"
#endif

#include "tapas/hot/exact_let.h"
#include "tapas/hot/oneside_insp2.h"

using tapas::debug::BarrierExec;

namespace {

#ifdef __CUDACC__

/**
 * \breif Atomic-add code from cuda-c-programming-guide (double precision version)
 */
__device__
static double _atomicAdd(double* address, double val) {
  // Should we use uint64_t ?
  static_assert(sizeof(unsigned long long int) == sizeof(double),   "sizeof(unsigned long long int) == sizeof(double)");
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
 * \breif Atomic-add code from cuda-c-programming-guide (double precision version)
 */
__device__
static float _atomicAdd(float* address, float val) {
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

#endif /* __CUDACC__ */

}



namespace tapas {

/**
 * @brief Provides MPI-based distributed SFC-based octree partitioning
 */
namespace hot {

using tapas::mpi::MPI_DatatypeTraits;

// fwd decl
template<class TSP> class Cell;

/**
 * @brief Remove redundunt elements in a std::vector. The vector must be sorted.
 *
 * This way is much faster than using std::set.
 */
template<class T, class Iterator>
std::vector<T> uniq(Iterator beg, Iterator end) {
  std::vector<T> v(beg, end);
  v.erase(unique(std::begin(v), std::end(v)), std::end(v));
  return v;
}

/**
 * @brief Returns the range of bodies from an array of T (body type) that belong to the cell specified by the given key.
 * @tparam BT Body type. (might be replaced by Iter::value_type)
 * @tparam Iter Iterator type of the body array.
 * @tparam Functor Functor type that retrieves morton key from a body type value.
 * @return returns std::pair of (pos, len)
 */
template <class SFC, class BT, class Iter, class Functor>
std::pair<typename SFC::KeyType, typename SFC::KeyType>
GetBodyRange(const typename SFC::KeyType k,
             Iter beg, Iter end,
             Functor get_key = tapas::util::identity<BT>) {
  using KeyType = typename SFC::KeyType;

  // When used in Refine(), a cells has sometimes no body.
  // In this special case, just returns (0, 0)
  if (beg == end) return std::make_pair(0, 0);

  auto less_than = [get_key] (const BT &hn, KeyType k) {
    return get_key(hn) < k;
  };

  auto fst = std::lower_bound(beg, end, k, less_than); // first node
  auto lst = std::lower_bound(fst, end, SFC::GetNext(k), less_than); // last node

  assert(lst <= end);

  return std::make_pair(fst - beg, lst - fst); // returns (pos, nb)
}

/**
 * @brief std::vector version of GetBodyRange
 */
template<class SFC, class T, class Functor>
std::pair<typename SFC::KeyType, typename SFC::KeyType>
GetBodyRange(const typename SFC::KeyType k,
             const std::vector<T> &hn,
             Functor get_key = tapas::util::identity<T>) {
  using Iter = typename std::vector<T>::const_iterator;
  return GetBodyRange<SFC, T, Iter, Functor>(k, hn.begin(), hn.end(), get_key);
}
template <class TSP>
struct HelperNode {
  using KeyType = typename TSP::SFC::KeyType;
  KeyType key;          //!< SFC key (Default: Morton)
  Vec<TSP::Dim, int> anchor; //!< SFC key-like vector without depth information
  index_t p_index;      //!< Index of the corresponding body
  index_t np;           //!< Number of particles in a node
};

template <class TSP>
std::vector<HelperNode<TSP>>
CreateInitialNodes(const typename TSP::Body *p, index_t np, const Region<TSP::Dim, typename TSP::FP> &r);

template <int DIM, class KeyType, class T>
void AppendChildren(KeyType k, T &s);

template <class TSP>
void SortBodies(const typename TSP::Body *b, typename TSP::Body *sorted,
                const HelperNode<TSP> *nodes,
                tapas::index_t nb);

template <class TSP>
void CompleteRegion(typename TSP::SFC x, typename TSP::SFC y, typename TSP::KeyVector &s);

template <class TSP>
index_t GetBodyNumber(const typename TSP::SFC k, const HelperNode<TSP> *hn,
                      index_t offset, index_t len);


template <class TSP> class Partitioner;

template<class T>
using uset = std::unordered_set<T>;

// new Traverse
template<class TSP>
void ReportInteractionType(typename Cell<TSP>::KeyType trg_key,
                           typename Cell<TSP>::KeyType src_key,
                           IntrFlag by_pred, IntrFlag orig) {
  using SFC = typename Cell<TSP>::SFC;

  tapas::debug::DebugStream e("check");
  e.out() << SFC::Simplify(trg_key) << " - " << SFC::Simplify(src_key) << "  ";

  e.out() << "Pred: ";
  e.out() << by_pred.ToString() << std::endl;

  e.out() << " Orig: ";
  e.out() << orig.ToString() << std::endl;
  
  e.out() << " " << (by_pred == orig ? "OK" : "NG") << std::endl;
}

/**
 * @brief Create an array of HelperNode from bodies
 * In the first stage of tree construction, one HelperNode is create for each body.
 * @return Array of HelperNode
 * @param bodies Pointer to an array of bodies
 * @param nb Number of bodies (length of bodies)
 * @param r Region object
 */
template <class TSP>
std::vector<HelperNode<TSP>> CreateInitialNodes(const typename TSP::Body *bodies,
                                                index_t nb,
                                                const Region<TSP::Dim, typename TSP::FP> &r) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;
    typedef typename TSP::SFC SFC;
    typedef HelperNode<TSP> HN;

    std::vector<HN> nodes(nb);
    FP num_cell = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension
    Vec<Dim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < Dim; ++d) {
      pitch[d] = (r.max()[d] - r.min()[d]) / num_cell;
    }

    for (index_t i = 0; i < nb; ++i) {
        // First, create 1 helper cell per particle
        HN &node = nodes[i];
        node.p_index = i;
        node.np = 1;

        // Particle pos offset is the offset of each coordinate value (x,y,z) in body structure
        Vec<Dim, FP> off = ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(bodies[i]));
        off -= r.min(); // set the base 0
        off /= pitch;   // quantitize offsets

        // Now 'off' is a Dim-dimensional index of the finest-level cell to which the particle belong.
        for (int d = 0; d < Dim; ++d) {
            node.anchor[d] = (int)(off[d]);
            // assume maximum boundary is inclusive, i.e., a particle can be
            // right at the maximum boundary.
            if (node.anchor[d] == (1 << SFC::MAX_DEPTH)) {
                TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
                node.anchor[d]--;
            }
        }
#ifdef TAPAS_DEBUG
        TAPAS_ASSERT(node.anchor >= 0);
# if 1
        if (!(node.anchor < (1 << SFC::MAX_DEPTH))) {
            TAPAS_LOG_ERROR() << "Anchor, " << node.anchor
                              << ", exceeds the maximum depth." << std::endl
                              << "Particle at "
                              << ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(bodies[i]))
                              << std::endl;
            TAPAS_DIE();
        }
# else
        assert(node.anchor < (1 << SFC::MAX_DEPTH));
# endif
#endif // TAPAS_DEBUG

        node.key = SFC::CalcFinestKey(node.anchor);
    }

    return nodes;
}

template <class TSP>
void SortBodies(const typename TSP::Body *b, typename TSP::Body *sorted,
                const HelperNode<TSP> *sorted_nodes,
                tapas::index_t nb) {
    for (index_t i = 0; i < nb; ++i) {
        sorted[i] = b[sorted_nodes[i].p_index];
    }
}

template <int DIM, class SFC, class T>
void AppendChildren(typename SFC::KeyType x, T &s) {
  using KeyType = typename SFC::KeyType;

  int x_depth = SFC::GetDepth(x);
  int c_depth = x_depth + 1;
  if (c_depth > SFC::MAX_DEPTH) return;
  x = SFC::IncrementDepth(x, 1);
  for (int i = 0; i < (1 << DIM); ++i) {
    KeyType child_key = ((KeyType)i << ((KeyType::MAX_DEPTH - c_depth) * DIM +
                                        SFC::DEPTH_BIT_WIDTH));
    s.push_back(x | child_key);
    TAPAS_LOG_DEBUG() << "Adding child " << (x | child_key) << std::endl;
  }
}

template <class TSP>
void CompleteRegion(typename TSP::SFC::KeyType x,
                    typename TSP::SFC::KeyType y,
                    typename TSP::SFC::KeyVector &s) {
  typedef typename TSP::SFC SFC;
  typedef typename SFC::KeyType KeyType;

  KeyType fa = SFC::FinestAncestor(x, y);
  typename SFC::KeyList w;

  AppendChildren<TSP::Dim, KeyType>(fa, w);
  tapas::PrintKeys(w, std::cout);

  while (w.size() > 0) {
    KeyType k = w.front();
    w.pop_front();
    TAPAS_LOG_DEBUG() << "visiting " << k << std::endl;
    if ((k > x && k < y) && !SFC::IsDescendant(k, y)) {
      s.push_back(k);
      TAPAS_LOG_DEBUG() << "Adding " << k << " to output set" << std::endl;
    } else if (SFC::IsDescendant(k, x) || SFC::IsDescendant(k, y)) {
      TAPAS_LOG_DEBUG() << "Adding children of " << k << " to work set" << std::endl;
      AppendChildren<TSP>(k, w);
    }
  }
  std::sort(std::begin(s), std::end(s));
}
template<class TSP, class Funct, class...Args>
void GlobalUpwardTraversal(Cell<TSP> &c, Funct f, Args&&...args) {
  using KeyType = typename Cell<TSP>::KeyType;
  using SFC = typename Cell<TSP>::SFC;
  auto &data = c.data();
  KeyType k = c.key();

  // c must be in the global tree hash table.
  TAPAS_ASSERT(data.ht_gtree_.count(k) == 1);

  // There are only two cases:
  // 1.  All the children of the cell c are in the global tree.
  //     This means c is not global-leaf cell.
  // 2.  None of the children of the cell c is in the global tree.
  //     This means c is a global-leaf.

  if (data.gleaves_.count(k) > 0) {
    // the cell c is a global leaf. The attr value is already calculated
    // as a local root in its owner process.
    return;
  }

  TAPAS_ASSERT(data.ht_gtree_.count(k) > 0);

  // c is not a global leaf.
  size_t nc = c.nsubcells();
  for (size_t ci = 0; ci < nc; ci++) {
    KeyType chk = SFC::Child(c.key(), ci);
    Cell<TSP> *child = data.ht_gtree_.at(chk);
    GlobalUpwardTraversal(*child, f, std::forward(args)...);
  }

  f(c, std::forward(args)...);
}

#define LOG(k_, code) do {                            \
    using SFC = typename TSP::SFC;                    \
    typename SFC::KeyType k = (k_);                   \
    std::string ks = SFC::Simplify(k);                \
    std::string kd = SFC::Decode(k);                  \
    tapas::debug::DebugStream ds("upward");           \
    int d = SFC::GetDepth((c).key());                 \
    for (int i = 0; i < d; i++) { ds.out() << "  "; } \
    code;                                             \
    ds.out() << std::endl;                            \
  } while(0)


} // namespace hot

#ifdef _F
# warning "Tapas function macro _F is already defined. "                \
  "Maybe it is conflicting other libraries or you included incompatible tapas headers."
#endif

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */
template <class TSP>
ProductIterator<tapas::iterator::CellIterator<hot::Cell<TSP>>,
                tapas::iterator::CellIterator<hot::Cell<TSP>>>
                                    Product(hot::Cell<TSP> &c1,
                                            hot::Cell<TSP> &c2) {
  typedef hot::Cell<TSP> CellType;
  typedef CellIterator<CellType> CellIterType;
  return ProductIterator<CellIterType, CellIterType>(
      CellIterType(c1), CellIterType(c2));
}

// New Tapas Static Params base class (refereed as TSP in classes)
template<int _DIM, class _FP, class _BODY_TYPE, size_t _BODY_COORD_OFST, class _BODY_ATTR, class _CELL_ATTR, class _KEY_TYPE=uint64_t>
struct HOT {
  static const constexpr int Dim = _DIM;
  static const constexpr size_t kBodyCoordOffset = _BODY_COORD_OFST;
  using FP = _FP;
  using Body = _BODY_TYPE;
  using BodyAttr = _BODY_ATTR;
  using CellAttr = _CELL_ATTR;
  using SFC = tapas::sfc::Morton<_DIM, _KEY_TYPE>;
  using Threading = tapas::threading::Default;

#ifdef __CUDACC__
  using Vectormap = tapas::Vectormap_CUDA_Packed<_DIM, _FP, _BODY_TYPE, _BODY_ATTR, _CELL_ATTR>;
  
  template<class T>
  using Allocator = typename Vectormap::template um_allocator<T>;
  
  template<class _CELL, class _BODY, class _INSP2, class _INSP1>
  using Mapper = hot::GPUMapper<_CELL, _BODY, _INSP2, _INSP1>;
  
#else
  
  using Vectormap = tapas::Vectormap_CPU<_DIM, _FP, _BODY_TYPE, _BODY_ATTR, _CELL_ATTR>;
  
  template<class T>
  using Allocator = std::allocator<T>;
  
  template<class _CELL, class _BODY, class _INSP2, class _INSP1>
  using Mapper = hot::CPUMapper<_CELL, _BODY, _INSP2, _INSP1>;
#endif

  template <class _TSP> using Partitioner = hot::Partitioner<_TSP>;
};

template<class _TSP>
struct Tapas {
  using TSP = _TSP;
  static const constexpr int Dim = TSP::Dim;
  static const constexpr size_t BodyCoordOffset = TSP::kBodyCoordOffset;
  using FP = typename TSP::FP;
  using Partitioner = typename TSP::template Partitioner<TSP>;
  using Region = tapas::Region<Dim, FP>;
  using Cell = hot::Cell<TSP>;
  using CellAttr = typename Cell::CellAttr;
  using Data = typename Cell::Data;
  using BodyIterator = typename Cell::BodyIterator;
  using Body = typename TSP::Body;
  using BodyAttr = typename TSP::BodyAttr;

  // Proxy cell for one-side traverse
  using ProxyCell1 = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::OnesideTraversePolicy<Dim, FP, Data>>;
  using ProxyAttr1 = tapas::hot::proxy::ProxyAttr<ProxyCell1>;
  using ProxyBody1 = tapas::hot::proxy::ProxyBody<Body, BodyAttr, typename ProxyCell1::Policy>;
  using ProxyBodyIterator1 = tapas::hot::proxy::ProxyBodyIterator<ProxyCell1>;

  // Proxy cell for two-side traversepolicy
  
  using ProxyCell2 = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;
  using ProxyAttr2 = tapas::hot::proxy::ProxyAttr<ProxyCell2>;
  using ProxyBody2 = tapas::hot::proxy::ProxyBody<Body, BodyAttr, typename ProxyCell2::Policy>;
  using ProxyBodyIterator2 = tapas::hot::proxy::ProxyBodyIterator<ProxyCell2>;

  using VecT = tapas::Vec<Dim, FP>;

  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(const Body *b, index_t nb, int max_nb,
                         MPI_Comm comm = MPI_COMM_WORLD) {
    TAPAS_ASSERT(b != nullptr);
    
    Partitioner part(max_nb);
    std::vector<BodyAttr> attrs(nb);
    memset(attrs.data(), 0, sizeof(attrs[0]) * attrs.size());
    
    return part.Partition(nullptr, b, attrs.data(), nullptr, nb, comm);
  }
  
  static Cell *Partition(const Body *b, const BodyAttr *a, index_t nb, int max_nb,
                         MPI_Comm comm = MPI_COMM_WORLD) {
    TAPAS_ASSERT(b != nullptr);
    TAPAS_ASSERT(a != nullptr);
    
    Partitioner part(max_nb);
    return part.Partition(nullptr, b, a, nullptr, nb, comm);
  }
  
  static Cell *Partition(const std::vector<Body> &bodies, index_t max_nb,
                         MPI_Comm comm = MPI_COMM_WORLD) {
    Partitioner part(max_nb);
    std::vector<BodyAttr> attrs(bodies.size());
    memset(attrs.data(), 0, sizeof(attrs[0]) * attrs.size());
    
    return part.Partition(nullptr, bodies.data(), attrs.data(), nullptr, bodies.size(), comm);
  }
  
  static Cell *Partition(const std::vector<Body> &bodies, const std::vector<BodyAttr> &attrs,
                         index_t max_nb, MPI_Comm comm = MPI_COMM_WORLD) {
    TAPAS_ASSERT(bodies.size() == attrs.size());
    Partitioner part(max_nb);
    return part.Partition(nullptr, bodies.data(), attrs.data(), nullptr, bodies.size(), comm);
  }

  /**
   * @brief Re-construct the tree and returns the new tree.
   * the argument pointer is deleted.
   */
  static Cell *Partition(Cell *root, int max_nb) {
    // Put weight values to each bodies by 'downward' traversal of the tree
#ifdef TAPAS_USE_WEIGHT
    PropagateWeight(root);
#endif
    
    // All cells are to be deleted.
    // all other data are recycled.
    Data *data = &(root->data());
    
    DestroyCells(root);

    std::vector<Body> bodies = std::move(data->local_bodies_);
    std::vector<BodyAttr> attrs = std::move(data->local_body_attrs_);

#ifdef TAPAS_USE_WEIGHT
    std::vector<double> weights = std::move(data->local_body_weights_);

    // Write local_body_weight_br and local_body_weight_lf to the performance report
    double wbr_sum = std::accumulate(data->local_body_weight_br_.begin(),
                                     data->local_body_weight_br_.end(),
                                     0);
    double wlf_sum = std::accumulate(data->local_body_weight_lf_.begin(),
                                     data->local_body_weight_lf_.end(),
                                     0);

    double w_sum = std::accumulate(weights.begin(), weights.end(), 0);

    data->time_rec_.Record(data->timestep_, "Weight-BR", wbr_sum);
    data->time_rec_.Record(data->timestep_, "Weight-LF", wlf_sum);
    data->time_rec_.Record(data->timestep_, "Weight", w_sum);
    
#else
    std::vector<double> weights = std::vector<double>(data->local_bodies_.size(), 1.0);
#endif

    data->ht_.clear();
    data->ht_let_.clear();
    data->ht_gtree_.clear();
    data->gleaves_.clear();
    data->lroots_.clear();
    data->let_used_key_.clear();
    data->local_br_.clear();
    data->leaf_keys_.clear();
    data->leaf_nb_.clear();
    data->leaf_owners_.clear();
    data->local_bodies_.clear();
    data->local_body_attrs_.clear();
    data->let_body_attrs_.clear();
    data->local_body_keys_.clear();
    data->proc_first_keys_.clear();
#ifdef TAPAS_USE_WEIGHT
    data->local_body_weights_.clear();
    data->local_body_weight_br_.clear();
    data->local_body_weight_lf_.clear();
#endif

    Partitioner part(max_nb);
    return part.Partition(data, bodies.data(), attrs.data(), weights.data(), bodies.size(), data->mpi_comm_);
  }

  static void Destroy(Cell *&root) {
    DestroyTree(root);
    root = nullptr;
  }

  template<class Funct, class...Args>
  static inline void Map(Funct f,
                         tapas::iterator::Bodies<Cell> bodies,
                         Args&&... args) {
    bodies.cell().mapper().Map(f, bodies, std::forward<Args>(args)...);
  }

  template<class Funct, class T1_Iter, class T2_Iter, class...Args>
  static inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args&&...args) {
    if (prod.size() > 0) {
      auto &cell = *(prod.t1_);  // "Cell" may be Cell or ProxyCell
      cell.mapper().MapP2(f, prod, std::forward<Args>(args)...);
    }
  }
  
  template <class Funct, class T1_Iter, class...Args>
  static inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args&&...args) {
    //TAPAS_LOG_DEBUG() << "map product iterator size: "
    //                  << prod.size() << std::endl;

    if (prod.size() > 0) {
      auto &cell = prod.t1_.cell(); // cell may be Cell or ProxyCell
      cell.mapper().MapP1(f, prod, std::forward<Args>(args)...);
    }
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args&&...args) {
    iter.cell().mapper().Map(f, iter, std::forward<Args>(args)...);
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, tapas::iterator::SubCellIterator<ProxyCell2> iter, Args&&...args) {
    // NOTE two-side (full) traversal ProxyCell is used for a single-parameter Map.
    iter.cell().mapper().Map(f, iter, std::forward<Args>(args)...);
  }

  template <class Funct, class ...Args>
  static inline void Map(Funct f, BodyIterator iter, Args&&...args) {
    iter.cell().mapper().Map(f, iter, std::forward<Args>(args)...);
  }

  template <class Funct, class ...Args>
  static inline void Map(Funct f, ProxyBodyIterator1 iter, Args&&...args) {
    iter.cell().mapper().Map(f, iter, std::forward<Args>(args)...);
  }

  template <class Funct, class ...Args>
  static inline void Map(Funct f, ProxyBodyIterator2 iter, Args&&...args) {
    iter.cell().mapper().Map(f, iter, std::forward<Args>(args)...);
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, Cell &c, Args&&...args) {
    c.mapper().Map(f, c, std::forward<Args>(args)...);
  }

  template<typename T, typename ReduceFunc>
  static inline void Reduce(Cell &/*parent*/, const T& dst, const T& src, ReduceFunc f) {
    T& d = const_cast<T&>(dst);
    f(d, src);
  }

  template<typename T, typename ReduceFunc>
  static inline void Reduce(ProxyCell2 &cell, const T&, const T&, ReduceFunc) {
    //std::cout << "Reduce: mark 'modified' to cell " << cell.key()  << " [" << cell.depth() << "]" << std::endl;
    cell.MarkModified();
    // nop.
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const CellType &c1, const CellType &c2, DistanceType t) {
    return c1.dX(c2, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const CellType &c1, const Body &b, DistanceType t) {
    return c1.dX(b, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const Body &b, const CellType &c1, DistanceType t) {
    if (getenv("TAPAS_DEBUG")) {
      std::cout << "tapas::dX(Cell, RealBody) is called" << std::endl;
    }
    return c1.dX(b, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const CellType &c1, const ProxyBody1 &b, DistanceType t) {
    if (getenv("TAPAS_DEBUG")) {
      std::cout << "tapas::dX(Cell, ProxyBody1) is called" << std::endl;
    }
    return c1.dX(b, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const CellType &c1, const ProxyBody2 &b, DistanceType t) {
    if (getenv("TAPAS_DEBUG")) {
      std::cout << "tapas::dX(Cell, ProxyBody2) is called" << std::endl;
    }
    return c1.dX(b, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const ProxyBody1 &b, const CellType &c1, DistanceType t) {
    if (getenv("TAPAS_DEBUG")) {
      std::cout << "tapas::dX(ProxyBody, ProxyBody) is called" << std::endl;
    }
    return c1.dX(b, t);
  }

  template<class CellType, class DistanceType>
  static inline VecT dX(const ProxyBody2 &b, const CellType &c1, DistanceType t) {
    if (getenv("TAPAS_DEBUG")) {
      std::cout << "tapas::dX(ProxyBody, ProxyBody) is called" << std::endl;
    }
    return c1.dX(b, t);
  }

  static inline VecT dx(const Body &b1, const Body &b2) {
    Vec<Dim, FP> pos1 = ParticlePosOffset<Dim, FP, TSP::BodyCoordOffset>::vec(reinterpret_cast<const void*>(&b1));
    Vec<Dim, FP> pos2 = ParticlePosOffset<Dim, FP, TSP::BodyCoordOffset>::vec(reinterpret_cast<const void*>(&b2));
    return (pos1 - pos2);
  }

  template<class CellType, class DistanceType>
  static inline FP Distance2(const CellType &c1, const CellType &c2, DistanceType t) {
    return dX(c1, c2, t).norm();
  }

  template<class CellType, class DistanceType>
  static inline FP Distance2(const CellType &c1, const Body &b, DistanceType t) {
    return dX(c1, b, t).norm();
  }

  template<class CellType, class DistanceType>
  static inline FP Distance2(const Body &b, const CellType &c1, DistanceType t) {
    return dX(c1, b, t).norm();
  }

  // Distance of ProxyBody(1) - Cell
  template<class CellType, class DistanceType>
  static inline FP Distance2(const CellType &c1, const ProxyBody1 &b, DistanceType t) {
    std::cout << __FILE__ << ":" << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;
    return dX(c1, b, t).norm();
  }

  // Distance of ProxyBody(1) - Cell
  template<class CellType, class DistanceType>
  static inline FP Distance2(const ProxyBody1 &b, const CellType &c1, DistanceType t) {
    return dX(c1, b, t).norm();
  }

  // Distance of ProxyBody(2) - Cell
  template<class CellType, class DistanceType>
  static inline FP Distance2(const CellType &c1, const ProxyBody2 &b, DistanceType t) {
    return dX(c1, b, t).norm();
  }

  // Distance of ProxyBody(2) - Cell
  template<class CellType, class DistanceType>
  static inline FP Distance2(const ProxyBody2 &b, const CellType &c1, DistanceType t) {
    return dX(c1, b, t).norm();
  }

  static inline FP Distance2(const Body &b1, const Body &b2) {
    Vec<Dim, FP> pos1 = ParticlePosOffset<Dim, FP, TSP::kBodyCoordOffset>::vec(reinterpret_cast<const void*>(&b1));
    Vec<Dim, FP> pos2 = ParticlePosOffset<Dim, FP, TSP::kBodyCoordOffset>::vec(reinterpret_cast<const void*>(&b2));
    return (pos1 - pos2).norm();
  }

#ifdef __CUDACC__

  __device__ __host__
  static void Reduce(Body &, double &to_val, double val) {
#ifdef __CUDA_ARCH__
    //printf("_atomicAdd() (dobule) is called.\n");
    _atomicAdd(&to_val, val);
#else
    //printf("to_val += val  (double) is called.\n");
    to_val += val;
#endif
  }

  __device__ __host__
  static void Reduce(Body &, float &to_val, float val) {
#ifdef __CUDA_ARCH__
    //printf("_atomicAdd() (float) is called.\n");
    _atomicAdd(&to_val, val);
#else
    //printf("to_val += val (float) is called.\n");
    to_val += val;
#endif
  }

  // host-only Accumulate.
  // In ExaFMM on Tapas, it is used for P2M on CPU.
  template<class T, typename ReduceFunc>
  __host__
  static void Reduce(Body &, T& to_val, T val, ReduceFunc f) {
    f(to_val, val);
  }

#else // not __CUDACC__

  template<typename T, typename ReduceFunc>
  static void Reduce(Body &, T& to_val, T val, ReduceFunc f) {
    f(to_val, val);
  }

  static void Reduce(Body &, double& to_val, double val) {
    to_val += val;
  }

  static void Reduce(Body &, float& to_val, float val) {
    to_val += val;
  }

#endif  // __CUDACC__

}; // struct Tapas
} // namespace tapas

#ifdef TAPAS_DEBUG_DUMP
template<class TSP>
std::ostream& operator<<(std::ostream& os, tapas::hot::Cell<TSP> &cell) {
  using CellType = tapas::hot::Cell<TSP>;
  using SFC = typename CellType::SFC;

  os << "Cell: " << "key     = " << cell.key() << std::endl;
  os << "      " << "        = " << SFC::Decode(cell.key()) << std::endl;
  os << "      " << "        = " << SFC::Simplify(cell.key()) << std::endl;
  os << "      " << "IsLeaf  = " << cell.IsLeaf() << std::endl;
  os << "      " << "IsLocal = " << cell.IsLocal() << std::endl;
  if (cell.IsLeaf()) {
    os << "      " << "nb      = " << cell.nb() << std::endl;
  } else {
    os << "      " << "nb      = " << "N/A" << std::endl;
  }
  return os;
}
#endif

#ifdef __CUDACC__
#define TAPAS_KERNEL __host__ __device__ __forceinline__
#else
#define TAPAS_KERNEL
#endif

#endif // TAPAS_HOT_
