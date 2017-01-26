#ifndef TAPAS_HOT_CELL_H__
#define TAPAS_HOT_CELL_H__

#include "tapas/hot/buildtree.h"
#include "tapas/hot/global_tree.h"
#include "tapas/hot/insp1.h"
#include "tapas/hot/mapper.h"
#include "tapas/hot/report.h"
#include "tapas/hot/shared_data.h"
#include "tapas/hot/exact_let.h"
#include "tapas/hot/oneside_insp2.h"
#include "tapas/iterator.h"

namespace {
namespace iter = tapas::iterator;
}

namespace tapas {
namespace hot {

template<class TSP>
void FindLocalRoots(typename Cell<TSP>::KeyType,
                    const typename Cell<TSP>::CellHashTable&,
                    typename Cell<TSP>::KeySet&);

template<class TSP> class Partitioner;

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

  friend struct ExactInsp2<TSP>;
  using Inspector2 = ExactInsp2<TSP>;
  using Inspector2_2 = OnesideInsp2<TSP>;

  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using SFC = typename TSP::SFC;
  using Reg = Region<Dim, FP>;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using VecT = tapas::Vec<Dim, FP>;

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

  template<class Funct, class...Args> static void DownwardMap(Funct f, Cell<TSP> &c, Args&&...args);

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
  VecT center() const {
    return center_;
  }

  VecT width() const {
    return region_.width();
  }

  FP width(int i) const {
    return region_.width(i);
  }

  const Reg& GetRegion() const {
    return region_;
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

  /**
   * Returns const reference to the local bodies
   */
  const std::vector<Body> &GetBodies() const {
    return data_->local_bodies_;
  }
  
  /**
   * Returns const reference to the local bodies
   */
  const std::vector<BodyAttr> &GetBodyAttrs() const {
    return data_->local_body_attrs_;
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

  static VecT CalcCenter(KeyType key, const Reg& region) {
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
#ifdef TAPAS_DEBUG
    if (IsLeaf()) {
      TAPAS_ASSERT(!"ERROR: Calling subcells() to a non-leaf cell.");
    }
#endif
    return SubCellIterator(*this);
  }

  SubCellIterator subcells() const {
#ifdef TAPAS_DEBUG
    if (IsLeaf()) {
      TAPAS_ASSERT(!"ERROR: Calling subcells() to a non-leaf cell.");
    }
#endif
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

  // cell, center
  inline VecT dX(const Cell &rhs, tapas::CenterClass) const {
    return (center() - rhs.center());
  }

  // cell-to-cell distance, shortest
  inline VecT dX(const Cell &rhs, tapas::ShortestClass) const {
    VecT dx = {0.0};
    for (int d = 0; d < Dim; d++) {
      // this cell's region of d-th dimension = a
      FP a_max = region_.max(d), a_min = region_.min(d);
      // rhs's region of d-th dimension = b
      FP b_max = rhs.region_.max(d), b_min = rhs.region_.min(d);

      if ((b_min <= a_min && a_min <= b_max)
          || (b_min <= a_max && a_max <= b_max)) {
        // the two regions overlap
        dx[d] = 0;
      } else if (a_min <= b_min && b_max <= a_max) {
        // a includes b
        dx[d] = 0;
      } else if (b_min <= a_min && a_max <= b_max) {
        // b includes a
        dx[d] = 0;
      } else if (a_max < b_min) {
        // the two regions are discrete (b is over a)
        dx[d] = b_min - a_max;
      } else if (a_min > b_max) {
        // the two regions are discrete (a is over b)
        dx[d] = a_min - b_max;
      } else {
        assert(0); // should not reach
      }
    }
    return dx;
  }

  // body, center
  inline VecT dX(const Body &b, tapas::CenterClass) const {
    VecT pos = ParticlePosOffset<Dim, FP, TSP::kBodyCoordOffset>::vec(reinterpret_cast<const void*>(&b));
    return (center() - pos);
  }

  // body, shortest
  inline VecT dX(const Body &b, tapas::ShortestClass) const {
    VecT pos = ParticlePosOffset<Dim, FP, TSP::kBodyCoordOffset>::vec(reinterpret_cast<const void*>(&b));
    VecT dx = {0.0};
    for (int d = 0; d < Dim; d++) {
      if ((region_.min(d) <= pos[d]) && (pos[d] <= region_.max(d))) {
        // the position[d] is included in region[d]
        dx[d] = 0;
      } else if (pos[d] < region_.min(d)) {
        dx[d] = region_.min(d) - pos[d];
      } else if (pos[d] > region_.max(d)) {
        dx[d] = pos[d] - region_.max(d);
      } else {
        assert(0);
      }
    }
    return dx;
  }

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
  const VecT center_; //!< The center of the cell

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

  MPI_Barrier(MPI_COMM_WORLD);

  for (size_t i = 0; i < keys_recv.size(); i++) {
    KeyType key = keys_recv[i];

    TAPAS_ASSERT(gtree.count(key) == 1);
    gtree[key]->attr() = attr_recv[i];
    data.local_upw_results_[key] = attr_recv[i];
  }

  TAPAS_ASSERT(keys_recv.size() == attr_recv.size());
}



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
       << "Cell not found for key \n      "
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

    TAPAS_LOG_ERROR() << ss.str();
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
      data.local_body_weights_[bidx] = wbr + 0.0014296 * wlf; // 0.014296 is for Spherical kernel.
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

}
}

#endif // TAPAS_HOT_CELL_H__

