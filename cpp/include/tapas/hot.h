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

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/sfc_morton.h"
#include "tapas/threading/default.h"
#include "tapas/mpi_util.h"
#include "tapas/geometry.h"

#include "tapas/hot/shared_data.h"
#include "tapas/hot/buildtree.h"
#include "tapas/hot/global_tree.h"
#include "tapas/hot/report.h"
#include "tapas/hot/mapper.h"
#include "tapas/hot/insp1.h"

#ifdef TAPAS_ONESIDE_LET
# include "tapas/hot/oneside_let.h"
#else
# include "tapas/hot/exact_let.h"
#endif

#define DEBUG_SENDRECV

using tapas::debug::BarrierExec;

namespace {
namespace iter = tapas::iterator;
}

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

template<class TSP>
void FindLocalRoots(typename Cell<TSP>::KeyType,
                    const typename Cell<TSP>::CellHashTable&,
                    typename Cell<TSP>::KeySet&);

template <class TSP> // TapasStaticParams
class Cell {
  friend class SamplingOctree<TSP, typename TSP::SFC>;
  friend class GlobalTree<TSP>;
  friend class Partitioner<TSP>;
  friend class iter::BodyIterator<Cell>;

  //========================================================
  // Internal structures
  //========================================================

 public:
  using CellAttr = typename TSP::CellAttr;

 protected:

#ifdef TAPAS_USE_WEIGHT
  /**
   * \brief A wrapper of user's cell attribute class for weighted re-partitioning.
   *
   * If weighted re-partitioning is activated (by compile-time macro switching),
   * Tapas needs to decide when to increase weights of cell/bodies.
   * Basically it's "when heavy computation is done on the Cell", but it is not obvious
   * to Tapas because computation is included in user's functions and it's a blackbox fom Tapas.
   * Thus, Tapas uses CellAttrWrapper to hook writes to cell attributes.
   *
   * This class is unnecessary when weighted re-partitioning is NOT activated.
   */
  struct CellAttrWrapper : CellAttr {
    friend Cell<TSP>;
    
   protected:
    Cell<TSP> &c_;

    CellAttrWrapper(Cell<TSP> &c) : c_(c) {
      // zero-clear the CellAttr part of this.
      memset(this, 0, sizeof(CellAttr));
    }
    
   public:
    inline CellAttrWrapper &operator=(const CellAttrWrapper& rhs) {
      c_.WeightBr();
      ((CellAttr&)*this) = (const CellAttr&)rhs;
      return *this;
    }

    inline CellAttrWrapper &operator=(const CellAttr &rhs) {
      c_.WeightBr();
      ((CellAttr&)*this) = rhs;
      return *this;
    }
  };
#endif

  //========================================================
  // Typedefs
  //========================================================
 public: // public type usings

#ifdef TAPAS_ONESIDE_LET
  friend struct OptInsp2<TSP>;
  using Inspector2 = OptInsp2<TSP>;
#else
  friend struct ExactInsp2<TSP>;
  using Inspector2 = ExactInsp2<TSP>;
#endif

  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using SFC = typename TSP::SFC;
  using Reg = Region<Dim, FP>;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using Vec = tapas::Vec<Dim, FP>;

  using Data = SharedData<TSP, SFC>;
  using CellHashTable = std::unordered_map<KeyType, Cell*>;
  using KeySet = std::unordered_set<KeyType>;

  using BodyType = typename TSP::Body;
  using BodyAttrType = typename TSP::BodyAttr;
  using Threading = typename TSP::Threading;

  using Body = BodyType;
  using BodyAttr = BodyAttrType;
  using Inspector1 = Insp1<TSP>;
  using Mapper = typename TSP::template Mapper<CellType, Body, Inspector2, Inspector1>;

  using BodyIterator = iter::BodyIterator<Cell>;
  using SubCellIterator = iter::SubCellIterator<Cell>;

  friend void FindLocalRoots<TSP>(KeyType, const CellHashTable&, KeySet&);

  //========================================================
  // Constructors
  //========================================================
 public:

  static Cell *CreateRemoteCell(KeyType k, int nb, Data *data) {
    Cell *c = new Cell(k, data->region_, 0, 0);
    c->is_leaf_ = false;
    c->is_local_ = false; // This cell is
    c->is_local_subtree_ = false;
    c->nb_ = nb;
    c->data_ = data;

    return c;
  }

  /**
   * \brief Constuctor
   */
  Cell(KeyType key, const Reg &reg, index_t bid, index_t numbodies)
      : key_(key)
      , nb_(numbodies)
      , bid_(bid)
      , region_(CalcRegion(key, reg))
      , center_((region_.max() + region_.min()) / 2)
#ifdef TAPAS_USE_WEIGHT
      , attr_(*this) // pass this pointer to CellAttrWrapper's ctor (CellAttrWrapper is to manage weights)
      , weight_lf_(1.0)
      , weight_br_(1.0)
#endif
  {
#ifndef TAPAS_USE_WEIGHT
    memset((void*)&attr_, 0, sizeof(attr_));
#endif
  }
  

  Cell(const Cell &rhs) = delete; // copy constructor is not allowed.
  Cell(Cell&& rhs) = delete; // move constructor is neither allowed.

  ~Cell() throw() {
  }
  //: tapas::BasicCell<TSP>(region, bid, nb)

  //========================================================
  // Member functions
  //========================================================

 public:
  KeyType key() const { return key_; }
  
  double WeightLf() const { return weight_lf_; }
  double WeightLf(double w = 1.0) { weight_lf_ += w; return weight_lf_; }
  double WeightBr() const { return weight_br_; }
  double WeightBr(double w = 1.0) { weight_br_ += w; return weight_br_; }
  
  template <class T> bool operator==(const T &) const { return false; }
  bool operator==(const Cell &c) const;
  bool operator<(const Cell &c) const;

  bool IsRoot() const;
  bool IsLocalSubtree() const;

  template<class Funct, class...Args> static void DownwardMap(Funct f, Cell<TSP> &c, Args...args);

  inline CellType &cell() { return *this; }
  inline const CellType &cell() const { return *this; }

  /**
   * @brief Returns if the cell is a leaf cell
   */
  bool IsLeaf() const;
  void SetLeaf(bool);

  /**
   * @brief Returns if the cell is local.
   */
  bool IsLocal() const;

  /**
   * @brief Returns idx-th subcell.
   */
  Cell &subcell(int idx);

  /**
   *
   */
  Vec center() const {
    return center_;
  }

  Vec width() const {
    return region_.width();
  }

  FP width(int i) const {
    return region_.width(i);
  }

  /**
   * @brief Returns the parent cell if it's local.
   *
   * Returns a reference to the parent cell object of this cell.
   * In this HOT implementation, parent cell of a local cell is
   * always a local cell.
   */
  Cell &parent() const;

  int depth() const {
    return SFC::GetDepth(key_);
  }

  Data &data() { return *data_; }
  const Data &data() const { return *data_; }
  Data* data_ptr() { return data_; }

#ifdef DEPRECATED
  typename TSP::Body &particle(index_t idx) const {
    return body(idx);
  }
#endif

  // Accessor functions to bodies & body attributes
  BodyType &body(index_t idx);
  const BodyType &body(index_t idx) const;

  // Internal use only
  inline index_t body_offset() {
    // returns body offset in the local_bodies or let_bodies_
    return bid_;
  }

  inline BodyType *body_base_ptr() {
    if (is_local_) {
      return data_->local_bodies_.data();
    } else {
      return data_->let_bodies_.data();
    }
  }

  inline BodyAttrType *body_attr_base_ptr() {
    if (is_local_) {
      return this->data_->local_body_attrs_.data();
    } else {
      return this->data_->let_body_attrs_.data();
    }
  }

  BodyType &local_body(index_t idx);
  const BodyType &local_body(index_t idx) const;

  BodyIterator bodies() {
    return BodyIterator(*this);
  }

  BodyIterator bodies() const {
    return BodyIterator(const_cast<CellType&>(*this));
  }

  tapas::iterator::Bodies<Cell> Bodies() {
    return tapas::iterator::Bodies<Cell>(*this);
  }

  BodyAttrType &body_attr(index_t idx);
  const BodyAttrType &body_attr(index_t idx) const;

  //BodyAttrType *body_attrs();
  //const BodyAttrType *body_attrs() const;

  BodyAttrType &local_body_attr(index_t idx);
  const BodyAttrType &local_body_attr(index_t idx) const;

  BodyAttrType *local_body_attrs();
  const BodyAttrType *local_body_attrs() const;

  /**
   * \brief Get number of local particles that belongs to the cell (directly or indirectly)
   * This function is mainly for debugging or checking result and internal use.
   * It is not recommended to use local_nb() for your main computation.
   * because it exposes the underlying implementation details of Tapas runtime.
   * In addition, if the cell spans over multiple processes, lcoal_nb() counts only local bodies
   * and does not return the "true" number.
   */
  inline index_t local_nb() const {
    return local_nb_;
  }

  static const constexpr bool Inspector = false;

  inline index_t nb() const {
#ifdef TAPAS_DEBUG
    if (!this->IsLeaf()) {
      TAPAS_ASSERT(!"Cell::nb() is not allowed for non-leaf cells.");
    }
#endif

    return nb_;
  }

#ifdef TAPAS_USE_WEIGHT
  CellAttrWrapper &attr() {
    return attr_;
  }
#else
  CellAttr &attr() {
    return attr_;
  }
#endif

  const CellAttr &attr() const {
    return (const CellAttr&)attr_;
  }

  /**
   * \brief check if the cell is large enough to spawn tasks recursively
   */
  inline bool SpawnTask() const {
    // Todo
    return depth() < 4;
  }

  void static ExchangeGlobalLeafAttrs(CellHashTable &gtree, const KeySet &lroots);

  static Reg CalcRegion(KeyType key, const Reg& R) {
    auto ret = R;
    SFC::template CalcRegion<FP>(key, R.max(), R.min(), ret.max(), ret.min());
    return ret;
  }

  static Vec CalcCenter(KeyType key, const Reg& region) {
    auto r = CalcRegion(key, region);
    return r.min() + r.width() / 2;
  }

  void Report() const { tapas::hot::Report<Data>(*data_); }

#ifdef DEPRECATED
  typename TSP::BT_ATTR *particle_attrs() const {
    return body_attrs();
  }
#endif
  SubCellIterator subcells() {
    return SubCellIterator(*this);
  }

  SubCellIterator subcells() const {
    return SubCellIterator(const_cast<CellType&>(*this));
  }

  const Reg &region() const { return region_; }

  MPI_Comm GetOptMPIComm() const {
    return data_->mpi_comm_;
  }

  MPI_Comm SetOptMPIComm(MPI_Comm) {
    MPI_Comm prev = data_->mpi_comm_;
    data_->mpi_comm_;
    return prev;
  }

  Mapper& mapper() {
    return data_->mapper_;
  }
  const Mapper &mapper() const { return data_->mapper; }

  inline FP Distance(const Cell &rhs, tapas::CenterClass) const {
    return tapas::Distance<Dim, tapas::CenterClass, FP>::Calc(*this, rhs);
  }

  //inline FP Distance(Cell &rhs, tapas::Edge) {
  //  return tapas::Distance<tapas::Edge, FP>::Calc(*this, rhs);
  //}

 protected:
  // utility/accessor functions
  inline Cell *Lookup(KeyType k) const;

  /**
   * @brief Returns the number of subcells. This is 0 or 2^DIM in HOT algorithm.
   */
  size_t nsubcells() const;

  //========================================================
  // Member variables
  //========================================================
 protected:
  KeyType key_; //!< Key of the cell
  bool is_leaf_;
  Data* data_;

  index_t nb_; //!< number of bodies in the local process (not bodies under this cell).
  index_t local_nb_;
  index_t bid_;

  bool is_local_; //!< if it's a local cell or LET cell.
  bool is_local_subtree_; //!< If all of its descendants are local.

  const Reg region_; //!< Local region
  const Vec center_; //!< The center of the cell

  Mapper mapper_;

#ifdef TAPAS_USE_WEIGHT
  CellAttrWrapper attr_;
#else
  CellAttr attr_;
#endif

  double weight_lf_; // computing weight from leaf-leaf or leaf-branch computations
  double weight_br_; // computing weight from branch-branch computations

  void CheckBodyIndex(index_t idx) const;
}; // class Cell

template<class T>
using uset = std::unordered_set<T>;

template<class TSP>
std::string k2s(typename Cell<TSP>::KeyType k) {
  std::stringstream ss;
  using SFC = typename Cell<TSP>::SFC;
#if 0
  ss << SFC::Simplify(k) << " " << SFC::Decode(k);
#else
  ss << SFC::Simplify(k);
#endif
  return ss.str();
}

// new Traverse
template<class TSP>
void ReportSplitType(typename Cell<TSP>::KeyType trg_key,
                     typename Cell<TSP>::KeyType src_key,
                     SplitType by_pred, SplitType orig) {
  using SFC = typename Cell<TSP>::SFC;

  tapas::debug::DebugStream e("check");
  e.out() << SFC::Simplify(trg_key) << " - " << SFC::Simplify(src_key) << "  ";

  e.out() << "Pred:";
  switch(by_pred) {
    case SplitType::Approx:     e.out() << "Approx";     break;
    case SplitType::Body:       e.out() << "Body";       break;
    case SplitType::SplitLeft:  e.out() << "SplitLeft";  break;
    case SplitType::SplitRight: e.out() << "SplitRight"; break;
    case SplitType::None:       e.out() << "None:";      break;
    default: assert(0);
  }

  e.out() << " Orig:";
  switch(orig) {
    case SplitType::Approx:     e.out() << "Approx";     break;
    case SplitType::Body:       e.out() << "Body";       break;
    case SplitType::SplitLeft:  e.out() << "SplitLeft";  break;
    case SplitType::SplitRight: e.out() << "SplitRight"; break;
    case SplitType::None:       e.out() << "None:";      break;
    default: assert(0);
  }

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

/**
 * \brief Exchange cell attrs of global leaves
 * Used in upward traversal. After all local trees are traversed, exchange global leaves
 * between processes.
 */
template<class TSP>
void Cell<TSP>::ExchangeGlobalLeafAttrs(typename Cell<TSP>::CellHashTable &gtree,
                                        const typename Cell<TSP>::KeySet &lroots) {
  // data.gleaves_ is unnecessary?
  using KeyType = typename Cell<TSP>::KeyType;

  std::vector<KeyType> keys_send(lroots.begin(), lroots.end());
  std::vector<KeyType> keys_recv;
  std::vector<CellAttr> attr_send;
  std::vector<CellAttr> attr_recv;

  auto &data = gtree[0]->data();

  for(size_t i = 0; i < keys_send.size(); i++) {
    KeyType k = keys_send[i];
    TAPAS_ASSERT(data.ht_.count(k) == 1);
    attr_send.push_back((CellAttr&)(data.ht_[k]->attr()));
  }

  tapas::mpi::Allgatherv(keys_send, keys_recv, MPI_COMM_WORLD);
  tapas::mpi::Allgatherv(attr_send, attr_recv, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // const constexpr int RANK = 0;
  // if (data.mpi_rank_ == RANK) {
  //   std::cout << "M2M: in ExchangeGlobalLeaves:" << std::endl;
  //   std::cout << "M2M: -----------------------------" << std::endl;
  //   for (size_t i = 0; i < keys_recv.size(); i++) {
  //     std::cout << "M2M: "
  //               << keys_recv[i]
  //               << " M=" << ""
  //               << attr_recv[i].M << ""
  //               << std::endl;
  //   }
  //   std::cout << "M2M: -----------------------------" << std::endl;
  // }

  MPI_Barrier(MPI_COMM_WORLD);

  for (size_t i = 0; i < keys_recv.size(); i++) {
    KeyType key = keys_recv[i];

    TAPAS_ASSERT(gtree.count(key) == 1);
    gtree[key]->attr() = attr_recv[i];
    data.local_upw_results_[key] = attr_recv[i];

    // if (key == 4035225266123964417 && data.mpi_rank_ == RANK) {
    //   std::cout << "debug: ExchangeGlobalLeafAttrs(): " << "key= " << key << std::endl;
    //   std::cout << "debug: gtree.count(key) = " << gtree.count(key) << std::endl;
    //   std::cout << "debug: ht.count(key) = " << data.ht_.count(key) << std::endl;
    //   std::cout << "debug: ExchangeGlobalLeafAttrs(): " << "M= " << attr_recv[i].M << std::endl;
    //   std::cout << "debug: ExchangeGlobalLeafAttrs(): " << "M= " << gtree[key]->attr().M << std::endl;
    //   std::cout << "debug: ExchangeGlobalLeafAttrs(): " << "&M= " << &(gtree[key]->attr().M) << std::endl;
    // }
  }

  TAPAS_ASSERT(keys_recv.size() == attr_recv.size());
}

template<class TSP, class Funct, class...Args>
void GlobalUpwardTraversal(Cell<TSP> &c, Funct f, Args...args) {
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
    GlobalUpwardTraversal(*child, f, args...);
  }

  f(c, args...);
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

// template<class TSP>
// template<class Funct, class...Args>
// void Cell<TSP>::DownwardMap(Funct f, Cell<TSP> &c, Args...args) {
//   f(c, args...);

//   if (c.IsLeaf()) return;

//   auto &data = c.data();

//   for (auto && ch_key : SFC::GetChildren(c.key())) {
// #ifdef TAPAS_DEBUG
//     if (data.ht_gtree_.count(ch_key) > 0 && data.ht_.count(ch_key) > 0) {
//       TAPAS_ASSERT(data.ht_gtree_[ch_key] == data.ht_[ch_key]);
//     }
// #endif

//     if      (data.ht_gtree_.count(ch_key) > 0) DownwardMap(f, *data.ht_gtree_.at(ch_key), args...);
//     else if (data.ht_.count(ch_key)       > 0) DownwardMap(f, *data.ht_.at(ch_key), args...);
//   }
// }

template <class TSP>
bool Cell<TSP>::operator==(const Cell &c) const {
  return key_ == c.key_;
}

template <class TSP>
bool Cell<TSP>::operator<(const Cell<TSP> &c) const {
  return key_ < c.key_;
}

template <class TSP>
bool Cell<TSP>::IsRoot() const {
  return SFC::GetDepth(key_) == 0;
}

template <class TSP>
bool Cell<TSP>::IsLocalSubtree() const {
  return is_local_subtree_;
}


template <class TSP>
bool Cell<TSP>::IsLeaf() const {
  return is_leaf_;
}

template <class TSP>
void Cell<TSP>::SetLeaf(bool b) {
  is_leaf_ = b;
}

template <class TSP>
bool Cell<TSP>::IsLocal() const {
  return is_local_;
}

template <class TSP>
size_t Cell<TSP>::nsubcells() const {
  if (IsLeaf()) return 0;
  else return (1 << TSP::Dim);
}

template <class TSP>
Cell<TSP> &Cell<TSP>::subcell(int idx) {
#ifdef TAPAS_DEBUG
  if (IsLeaf()) {
    TAPAS_LOG_ERROR() << "Trying to access children of a leaf cell." << std::endl;
    TAPAS_DIE();
  }
#endif

  KeyType child_key = SFC::Child(key_, idx);
  Cell *c = Lookup(child_key);

#ifdef TAPAS_DEBUG
  // assert c != nullptr
  if (c == nullptr) {
    std::stringstream ss;
    ss << "In MPI rank " << data_->mpi_rank_ << ": "
       << "Cell not found for key "
       << SFC::Simplify(child_key) << " "
       << SFC::Decode(child_key) << " "
       << child_key
       << std::endl;
    ss << "In MPI rank " << data_->mpi_rank_ << ": Anscestors are:" << std::endl;

    for (KeyType k = key_; k != 0; k = SFC::Parent(k)) {
      ss << "      "
         << SFC::Simplify(k) << " "
         << SFC::Decode(k) << " "
         << k << " ";

      if (data_->lroots_.count(k) > 0) {
        ss << "[lroot] ";
      } else if (data_->ht_gtree_.count(k) > 0) {
        ss << "[gtree] ";
      }

      ss << std::endl;
    }

    TAPAS_LOG_ERROR() << ss.str(); abort();
    TAPAS_ASSERT(c != nullptr);
  }
#endif // TAPAS_DEBUG

  return *c;
}

template <class TSP>
inline Cell<TSP> *Cell<TSP>::Lookup(KeyType k) const {
  // Try the local hash.
  auto &ht = data_->ht_;
  auto &ht_let = data_->ht_let_;
  auto &ht_gtree = data_->ht_gtree_;

  auto i = ht.find(k);
  if (i != ht.end()) {
    assert(i->second != nullptr);
    return i->second;
  }

  i = ht_let.find(k);
  // If the key is not in local hash, next try LET hash.
  if (i != ht_let.end()) {
    assert(i->second != nullptr);
    return i->second;
  }

  // NOTE: we do not search in ht_gtree.
  //       ht_gtree is for global tree, and used only in upward phase.
  //       If a cell exists only in ht_gtree, only the value of attr of the cell matters
  //       and whether the cell is leaf or not.
  //       Thus, cells in ht_gtree are not used in dual tree traversal.
  i = ht_gtree.find(k);
  // If the key is not in local hash, next try LET hash.
  if (i != ht_gtree.end()) {
    assert(i->second != nullptr);
    return i->second;
  }

  return nullptr;
}

template <class TSP>
Cell<TSP> &Cell<TSP>::parent() const {

#ifdef TAPAS_DEBUG
  // \todo Split TAPAS_DEBUG and TAPAS_DEV_DEBUG
  if (IsRoot()) {
    TAPAS_LOG_ERROR() << "Trying to access parent of the root cell." << std::endl;
    TAPAS_DIE();
  }
#endif

  KeyType parent_key = SFC::Parent(key_);
  auto *c = Lookup(parent_key);

#ifdef TAPAS_DEBUG
  if (c == nullptr) {
    TAPAS_LOG_ERROR() << "Parent (" << parent_key << ") of "
                      << "cell (" << key_ << ") not found.\n"
                      << "Parent key = " << SFC::Decode(parent_key) << "\n"
                      << "Child key =  " << SFC::Decode(key_)
                      << std::endl;
    TAPAS_DIE();
  }
#endif

  return *c;
}

template<class TSP>
static void DestroyCells(Cell<TSP> *root) throw() {
  using CellT = Cell<TSP>;

  std::set<CellT*> ptrs;

  auto &data = root->data();

  // Free all Cell pointers (except this).
  for (auto kv : data.ht_) {
    ptrs.insert(kv.second);
  }
  for (auto kv : data.ht_let_) {
    ptrs.insert(kv.second);
  }

  for (auto p : ptrs) {
    delete p;
  }
}

#ifdef TAPAS_USE_WEIGHT

namespace {

// A recursive subroutine for PropagateWeight() function.
template<class Cell>
void PropagateFunc(Cell *cell, typename Cell::Data &data) {
  using SFC = typename Cell::SFC;
  using KeyType = typename SFC::KeyType;
  
  if (cell->IsLeaf()) {
    // Put the leaf's weight on each body
    for (size_t bi = 0; bi < cell->nb(); bi++) {
      size_t bidx = cell->body_offset() + bi;
      double wbr = cell->WeightBr();
      double wlf = cell->WeightLf();
      //double alpha = 0.1;
      data.local_body_weight_br_[bidx] = wbr;
      data.local_body_weight_lf_[bidx] = wlf;
      data.local_body_weights_[bidx] = wbr + 0.014296 * wlf; // 0.014296 is for Spherical kernel.
    }
  } else {
    // Add the parent's weight to the children
    for (KeyType ck : SFC::GetChildren(cell->key())) {
      if (data.ht_.count(ck) > 0) {
        // the children (with key ck) is local
        Cell *child = data.ht_[ck];
        child->WeightBr(cell->WeightBr());
        PropagateFunc(child, data);
      }
    }
  }
}

} // anon namespace

template<class Cell>
void PropagateWeight(Cell *root) {
  using Data = typename Cell::Data;
  Data &data = root->data();

  data.local_body_weight_br_.resize(data.local_bodies_.size(), 1);
  data.local_body_weight_lf_.resize(data.local_bodies_.size(), 1);
  data.local_body_weights_.resize(data.local_bodies_.size(), 1);

  PropagateFunc(data.ht_[0], data);

#if 0
  std::cout << "------------ body weights -------------" << std::endl;
  for (size_t i = 0; i < data.local_body_weights_.size(); i++) {
    std::cout << i << " " << data.local_body_weights_[i] << std::endl;
  }
#endif

}

#endif

/**
 * \brief Destroy the tree completely.
 */
template<class TSP>
static void DestroyTree(Cell<TSP> *root) throw() {
  // Free memory
  using CellT = Cell<TSP>;

  std::set<CellT*> ptrs;

  auto &data = root->data();
  auto *pdata = &data;

  DestroyCells(root);

  // Free MPI_Datatypes
#ifdef USE_MPI
  MPI_Type_free(&data.mpi_type_key_);
  MPI_Type_free(&data.mpi_type_attr_);
  MPI_Type_free(&data.mpi_type_body_);
  MPI_Type_free(&data.mpi_type_battr_);
#endif

  // Delete SharedData structure
  delete pdata;
}

template <class TSP>
inline void Cell<TSP>::CheckBodyIndex(index_t idx) const {
  //TAPAS_ASSERT(this->nb() >= 0);
  (void)idx;

  // debug
  TAPAS_ASSERT(idx < this->nb());
  TAPAS_ASSERT(this->IsLeaf() && "body or body attribute access is not allowed for non-leaf cells.");

  if (is_local_) {
    TAPAS_ASSERT(bid_ + idx < data_->local_bodies_.size());
  } else {
    TAPAS_ASSERT(bid_ + idx < data_->let_bodies_.size());
  }
}

template <class TSP>
inline const typename TSP::Body &Cell<TSP>::body(index_t idx) const {
  CheckBodyIndex(idx);

  if (is_local_) {
    return data_->local_bodies_[bid_ + idx];
  } else {
    return data_->let_bodies_[bid_ + idx];
  }
}

template <class TSP>
inline typename TSP::Body &Cell<TSP>::body(index_t idx) {
  return const_cast<typename TSP::Body &>(const_cast<const Cell<TSP>*>(this)->body(idx));
}

template <class TSP>
const typename TSP::Body &Cell<TSP>::local_body(index_t idx) const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body() can be called only for local cells.");
  TAPAS_ASSERT((size_t)idx < data_->local_bodies_.size());

  // TODO is it correct?
  return data_->local_bodies_[bid_ + idx];
}

template <class TSP>
typename TSP::Body &Cell<TSP>::local_body(index_t idx) {
  return const_cast<typename TSP::Body &>(const_cast<const Cell<TSP>*>(this)->local_body(idx));
}

// template <class TSP>
// const typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
//   CheckBodyIndex(0);

//   if (is_local_) {
//     return data_->local_body_attrs_.data() + this->bid();
//   } else {
//     return data_->let_body_attrs_.data() + this->bid();
//   }
// }

// template <class TSP>
// typename TSP::BT_ATTR *Cell<TSP>::body_attrs() {
//   return const_cast<typename TSP::BT_ATTR &>(const_cast<const Cell<TSP>*>(this)->local_attrs());
// }

template <class TSP>
const typename TSP::BodyAttr &Cell<TSP>::body_attr(index_t idx) const {
  CheckBodyIndex(idx);

  if (is_local_) {
    return this->data_->local_body_attrs_[bid_ + idx];
  } else {
    return this->data_->let_body_attrs_[bid_ + idx];
  }
}

template <class TSP>
typename TSP::BodyAttr &Cell<TSP>::body_attr(index_t idx) {
  return const_cast<typename TSP::BodyAttr &>(const_cast<const Cell<TSP>*>(this)->body_attr(idx));
}

/**
 * \brief Returns a pointer to the first element of local bodies.
 * This function breaks the abstraction of Tapas and should be used only for
 * debugging / result checking purpose.
 */
template <class TSP>
const typename TSP::BodyAttr *Cell<TSP>::local_body_attrs() const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body_attrs() is only allowed for local cells");

  return data_->local_body_attrs_.data();
}

/**
 * \brief Non-const version of local_body_attrs()
 */
template <class TSP>
typename TSP::BodyAttr *Cell<TSP>::local_body_attrs() {
  return const_cast<typename TSP::BodyAttr *>(const_cast<const Cell<TSP>*>(this)->local_body_attrs());
}


/**
 * \brief Returns an attr of a body specified by idx.
 * This function breaks the abstraction of Tapas, thus should be used only for debugging purpose.
 */
template <class TSP>
const typename TSP::BodyAttr &Cell<TSP>::local_body_attr(index_t idx) const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body_attr(...) is allowed only for local cells.");
  TAPAS_ASSERT(idx < (index_t)data_->local_body_attrs_.size());

  return data_->local_body_attrs_[bid_ + idx];
}

/**
 * \brief Non-const version of Cell::local_body_attr()
 */
template <class TSP>
typename TSP::BodyAttr &Cell<TSP>::local_body_attr(index_t idx) {
  return const_cast<typename TSP::BodyAttr &>(const_cast<const Cell<TSP>*>(this)->local_body_attr(idx));
}

template <class TSP> // Tapas static params
class Partitioner {
 private:
  const index_t max_nb_;

  using BodyType = typename TSP::Body;
  using KeyType = typename Cell<TSP>::KeyType;
  using CellAttr = typename Cell<TSP>::CellAttr;
  using CellHashTable = typename Cell<TSP>::CellHashTable;

  using KeySet = typename Cell<TSP>::SFC::KeySet;

  using SFC = typename TSP::SFC;
  using HT = typename Cell<TSP>::CellHashTable;

  using Data = typename Cell<TSP>::Data;

 public:
  Partitioner(unsigned max_nb): max_nb_(max_nb) {}

  /**
   * @brief Partition the space and build the tree
   */
  Cell<TSP> *Partition(Data *data, BodyType *b, index_t nb, MPI_Comm comm);
  Cell<TSP> *Partition(Data *data, BodyType *b, const double *w, index_t nb, MPI_Comm comm);

  /**
   * @brief Overloaded version of Partitioner::Partition
   */
  Cell<TSP>* Partition(std::vector<BodyType> &b) {
    return Partition(b.data(), b.size());
  }


 public:
  //---------------------
  // Supporting functions
  //---------------------

  /**
   * @brief Find owner process from a head-key list.
   * The argument head_list contains SFC keys that are the first keys of processes.
   * head_list[P] is the first SFC key belonging to process P.
   * Because the first element is always 0 (by definition of space filling curve),
   * the result must be always >= 0.
   *
   */
  static int
  FindOwnerProcess(const std::vector<KeyType> &head_list, KeyType key) {
    TAPAS_ASSERT(Cell<TSP>::SFC::RemoveDepth(head_list[0]) == 0);
    auto comp = [](KeyType a, KeyType b) {
      return Cell<TSP>::SFC::RemoveDepth(a) < Cell<TSP>::SFC::RemoveDepth(b);
    };
    return std::upper_bound(head_list.begin(), head_list.end(), key, comp) - head_list.begin() - 1;
  }

  static std::vector<int>
  FindOwnerProcess(const std::vector<KeyType> &head_key_list,
                   const std::vector<KeyType> &keys) {
    std::vector<int> owners(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
      owners[i] = FindOwnerProcess(head_key_list, keys[i]);
    }

    return owners;
  }

  /**
   * \brief Select cells to be sent as a response in Insp2::Exchange
   * The request lists are made conservatively, thus not all the requested cells exist in the sender process.
   * Check the requested list and replace non-existing cells with existing cells by the their finest anscestors.
   * If attribute of a cell is requested but the cell is actually a leaf,
   * both of the attribut and body must be sent.
   */
  static void SelectResponseCells(std::vector<KeyType> &cell_attr_keys, std::vector<int> &attr_src_pids,
                                  std::vector<KeyType> &leaf_keys, std::vector<int> &leaf_src_pids,
                                  const HT& hash) {
    std::set<std::pair<int, KeyType>> res_attr; // keys (and their destinations) of which attributes will be sent as response.
    std::set<std::pair<int, KeyType>> res_body; // keys (and their destinations) of which bodies will be sent as response.

    TAPAS_ASSERT(cell_attr_keys.size() == attr_src_pids.size());
    TAPAS_ASSERT(leaf_keys.size() == leaf_src_pids.size());

    for (size_t i = 0; i < cell_attr_keys.size(); i++) {
      KeyType k = cell_attr_keys[i];
      int src_pid = attr_src_pids[i]; // PID of the process that requested k.

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      if (k == 0) {
        // This should not happend because if k is root node,
        // that means the process does not have any anscestor of the original k (except the root).
        // The requester sent the request to a wrong process
        TAPAS_ASSERT(false);
      }

      res_attr.insert(std::make_pair(src_pid, k));

      if (hash.at(k)->IsLeaf()) {
        res_body.insert(std::make_pair(src_pid, k));
      }
    }

    for (size_t i = 0; i < leaf_keys.size(); i++) {
      KeyType k = leaf_keys[i];
      int src_pid = leaf_src_pids[i];

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      TAPAS_ASSERT(k != 0); // the same reason above
      TAPAS_ASSERT(hash.count(k) > 0);
      TAPAS_ASSERT(hash.at(k)->IsLeaf());

      res_body.insert(std::make_pair(src_pid, k));
    }

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&res_attr, &res_body](int rank, int) {
        std::cerr << "Rank " << rank << " SelectResponseCells: keys_attr.size() = " << res_attr.size() << std::endl;
        std::cerr << "Rank " << rank << " SelectResponseCells: keys.body.size() = " << res_body.size() << std::endl;
      });
#endif

    // Set values to the vectors
    cell_attr_keys.resize(res_attr.size());
    attr_src_pids.resize(res_attr.size());

    int idx = 0;
    for (auto & iter : res_attr) {
      attr_src_pids[idx] = iter.first;
      cell_attr_keys[idx] = iter.second;
      idx++;
    }

    leaf_keys.resize(res_body.size());
    leaf_src_pids.resize(res_body.size());

    idx = 0;
    for (auto & iter : res_body) {
      leaf_src_pids[idx] = iter.first;
      leaf_keys[idx] = iter.second;

      idx++;
    }

    return;
  }

  static void KeysToAttrs(const std::vector<KeyType> &keys,
                          std::vector<CellAttr> &attrs,
                          const HT& hash) {
    // functor
    auto key_to_attr = [&hash](KeyType k) -> CellAttr& {
      return hash.at(k)->attr();
    };
    
    attrs.resize(keys.size());
    std::transform(keys.begin(), keys.end(), attrs.begin(), key_to_attr);
  }

  static void KeysToBodies(const std::vector<KeyType> &keys,
                           std::vector<index_t> &nb,
                           std::vector<BodyType> &bodies,
                           const HT& hash) {
    nb.resize(keys.size());
    bodies.clear();

    // In BH, each leaf has 0 or 1 body (while every cell has attribute)
    for (size_t i = 0; i < keys.size(); i++) {
      KeyType k = keys[i];
      auto *c = hash.at(k);
      nb[i] = c->IsLeaf() ? c->nb() : 0;

      for (size_t bi = 0; bi < nb[i]; bi++) {
        bodies.push_back(c->body(bi));
      }
    }
  }

}; // class Partitioner

template<class TSP>
Cell<TSP>*
Partitioner<TSP>::Partition(typename Cell<TSP>::Data *data,
                            typename TSP::Body *b,
                            index_t num_bodies,
                            MPI_Comm comm) {
  // If `w` is omitted, use nullptr instead.
  // In the first tree construction, there is no information of body weights
  // so a vector of 1.0 is used inside SamplingOctree class.
  return Partition(data, b, nullptr, num_bodies, comm);
}

/**
 * @brief Partition the simulation space and build SFC key based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param numBodies Length of b (NOT the total number of bodies over all processes)
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 * @todo In this function keys are exchanged using alltoall communication, as well as bodies.
 *       In extremely large scale systems, calculating keys locally again after communication
 *       might be faster.
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename Cell<TSP>::Data *data,
                            typename TSP::Body *b,
                            const double *w,
                            index_t num_bodies,
                            MPI_Comm comm) {
  using SFC = typename TSP::SFC;
  using CellType = Cell<TSP>;
  using Data = typename CellType::Data;

  if (data == nullptr) {
    // First timestep
    data = new Data(comm);
    data->ncrit_ = max_nb_;
    data->sample_rate_ = SamplingOctree<TSP, SFC>::SamplingRate();
  } else {
    // if `data` is not NULL,
    // This is re-partitioning. Increase time step counter
    // for reporting.
    data->timestep_++;
  }

  // Build local trees
  SamplingOctree<TSP, SFC> stree(b, w, num_bodies, data, max_nb_);
  stree.Build();

  // Build Global trees
  GlobalTree<TSP>::Build(*data);

#ifdef TAPAS_DEBUG_DUMP
  {
    tapas::debug::DebugStream e("cells");

    for (auto&& iter : data->ht_) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      e.out() << SFC::Simplify(k) << " "
              << "d=" << SFC::GetDepth(k) << " "
              << "leaf=" << c->IsLeaf() << " "
          //<< "owners=" << std::setw(2) << std::right << 0 << " "
              << "nb=" << std::setw(3) << (c->IsLeaf() ? (int)c->nb() : -1) << " "
              << "center=[" << c->center() << "] "
          //<< "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
          //<< "parent=" << SFC::Simplify(SFC::Parent(k))  << " "
              << std::endl;
    }
  }
#endif

  // Initialize the mapper class (mainly for GPU)
  data->mapper_.Setup();

  // return the root cell (root key is always 0)
  return data->ht_[0];
}

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
template<int _DIM, class _FP, class _BODY_TYPE, size_t _BODY_COORD_OFST, class _BODY_ATTR, class _CELL_ATTR>
struct HOT {
  static const constexpr int Dim = _DIM;
  static const constexpr size_t kBodyCoordOffset = _BODY_COORD_OFST;
  using FP = _FP;
  using Body = _BODY_TYPE;
  using BodyAttr = _BODY_ATTR;
  using CellAttr = _CELL_ATTR;
  using SFC = tapas::sfc::Morton<_DIM, uint64_t>;
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
  using FP = typename TSP::FP;
  using Partitioner = typename TSP::template Partitioner<TSP>;
  using Region = tapas::Region<Dim, FP>;
  using Cell = hot::Cell<TSP>;
  using CellAttr = typename Cell::CellAttr;
  using BodyIterator = typename Cell::BodyIterator;
  using Body = typename TSP::Body;
  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP>;
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<TSP>;
  using ProxyBodyIterator = tapas::hot::proxy::ProxyBodyIterator<TSP>;
  using Data = typename Cell::Data;

  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(Body *b, index_t nb, int max_nb,
                         MPI_Comm comm = MPI_COMM_WORLD) {
    Partitioner part(max_nb);
    return part.Partition(nullptr, b, nb, comm);
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
    return part.Partition(data, bodies.data(), weights.data(), bodies.size(), data->mpi_comm_);
  }

  static void Destroy(Cell *&root) {
    DestroyTree(root);
    root = nullptr;
  }

  template<class Funct, class...Args>
  static inline void Map(Funct f,
                         tapas::iterator::Bodies<Cell> bodies,
                         Args... args) {
    bodies.cell().mapper().Map(f, bodies, args...);
  }

  template<class Funct, class T1_Iter, class T2_Iter, class...Args>
  static inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      auto &cell = *(prod.t1_);  // "Cell" may be Cell or ProxyCell
      cell.mapper().MapP2(f, prod, args...);
    }
  }

  template <class Funct, class T1_Iter, class...Args>
  static inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    TAPAS_LOG_DEBUG() << "map product iterator size: "
                      << prod.size() << std::endl;

    if (prod.size() > 0) {
      auto &cell = prod.t1_.cell(); // cell may be Cell or ProxyCell
      cell.mapper().MapP1(f, prod, args...);
    }
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    iter.cell().mapper().Map(f, iter, args...);
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, tapas::iterator::SubCellIterator<ProxyCell> iter, Args...args) {
    iter.cell().mapper().Map(f, iter, args...);
  }

  template <class Funct, class ...Args>
  static inline void Map(Funct f, tapas::iterator::BodyIterator<Cell> iter, Args...args) {
    iter.cell().mapper().Map(f, iter, args...);
  }

  template <class Funct, class ...Args>
  static inline void Map(Funct f, ProxyBodyIterator iter, Args...args) {
    iter.cell().mapper().Map(f, iter, args...);
  }

  template <class Funct, class...Args>
  static inline void Map(Funct f, Cell &c, Args...args) {
    c.mapper().Map(f, c, args...);
  }

  template<typename T, typename ReduceFunc>
  static inline void Reduce(Cell &/*parent*/, const T& dst, const T& src, ReduceFunc f) {
    T& d = const_cast<T&>(dst);
    f(d, src);
  }

  template<typename T, typename ReduceFunc>
  static inline void Reduce(ProxyCell &cell, const T&, const T&, ReduceFunc) {
    //std::cout << "Reduce: mark 'modified' to cell " << cell.key()  << " [" << cell.depth() << "]" << std::endl;
    cell.MarkModified();
    // nop.
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
