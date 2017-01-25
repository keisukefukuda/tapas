#ifndef TAPAS_HOT_PROXY_PROXY_CELL_H_
#define TAPAS_HOT_PROXY_PROXY_CELL_H_

#include "tapas/hot/proxy/proxy_attr.h"
#include "tapas/hot/proxy/proxy_body.h"
#include "tapas/hot/proxy/proxy_body_attr.h"
#include "tapas/hot/proxy/proxy_body_iterator.h"

#include "tapas/hot/let_common.h"

namespace tapas {
namespace hot {

template<class TSP> class Cell;

namespace proxy {

template<class PROXY_CELL> struct ProxyMapper;

/**
 * \brief ProxyCell
 * \tparam _TSP Tapas static params
 * \tparam _POLICY Abstraction of the coordination/distance system of the cell (see below)
 *
 * ProxyCell class provides features for inspectors. ProxyCell objects mimic ordinary 
 * Cell objects but behaves differently.
 * First, recursive Map() calls to its children or bodies are not actually recursive.
 * Instead, Map() calls are hooked and recorded to determine if the cell is split or not
 * (i.e. the two cells are 'far' or 'near').
 * Also, writes/updates to cell attributes are inactivated by overloading operator=().
 * 
 * About the Policy class
 * In Tapas and especially its LET construction system, distances between cells are abstracted.
 *
 * For example, the OneSideTraversePolicy used in one-side LET inspector has only a region, width, 
 * and level of the pseudo cell but no concrete coordinate of a cell. 
 * A distance between such pseudo cells are the shortest distance between the two regions.
 * 
 * Some functions including dX() and Distance() are delegated to the policy class.
 * dX() and Distance() are always provided , but some other functions such as center() are optional.
 * 
 */
template<class _TSP, class _POLICY>
class ProxyCell : public _POLICY {
  friend ProxyAttr<ProxyCell>;

  using Base = _POLICY;

 public:
  using Policy = _POLICY;

  using TSP = _TSP;
  static const constexpr bool Inspector = true;
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;

  using RealCellType = Cell<TSP>;
  using CellType = RealCellType;
  using RealBody = typename TSP::Body;
  using RealBodyAttr = typename TSP::BodyAttr;
  
  using CellAttr = ProxyAttr<ProxyCell>;
  using Body = ProxyBody<RealBody, RealBodyAttr, Policy>;
  using BodyAttr = ProxyBodyAttr<RealBody, RealBodyAttr>;
  Body& local_body(int) { static Body b; return b; }

  using KeyType = typename tapas::hot::Cell<TSP>::KeyType;
  using SFC = typename tapas::hot::Cell<TSP>::SFC;
  using Data = typename CellType::Data;
  using Reg = Region<Dim, FP>;

  using Threading = typename CellType::Threading;

  using Mapper = ProxyMapper<ProxyCell>;
  using VecT = typename CellType::VecT;

  // ctor
  template<class...Args>
  ProxyCell(const Data &data, int *clock, Args...args)
      : Base(data, args...), data_(data)
      , marked_attr_(false), marked_split_(false), marked_nb_(false), marked_modified_(false), clock_(clock)
      , is_local_(false), attr_(this), children_()
  {
    CellType *c = this->Base::RealCell();
    if (c != nullptr) {
      attr_ = ProxyAttr<ProxyCell>(this, c->attr());
    }
  }

  ~ProxyCell() {
    if (children_.size() > 0) {
      for (ProxyCell *ch : children_) {
        if (ch != nullptr) {
          delete ch;
        }
      }
      children_.clear();
    }
  }
  
  void ClearFlags() {
    marked_attr_ = false;
    marked_split_ = false;
    marked_nb_ = false;
    marked_modified_ = false;
  }
  
  inline ProxyCell &cell() { return *this; }
  inline const ProxyCell &cell() const { return *this; }

  inline Mapper &mapper() { return mapper_; }
  inline const Mapper &mapper() const { return mapper_; }

  // dummy
  double Weight() const { return 0; }

  /**
   * bool ProxyCell::operator==(const ProxyCell &rhs) const
   */
  bool operator==(const ProxyCell &rhs) const {
    return (Base&)*this == (Base&)rhs;
  }

  bool operator<(const ProxyCell &rhs) const {
    return (const Base&)*this < (const Base&)rhs;
  }

  template<class UserFunct, class...Args>
  static IntrFlag PredSplit2(ProxyCell &trg_cell, ProxyCell &src_cell, UserFunct f, Args...args) {
    f(trg_cell, src_cell, args...);

    if (tapas::mpi::Rank() == 0) {
      std::cout << "PredSplit2: " << trg_cell.depth() << " " << src_cell.depth() << std::endl;
    }
    
    IntrFlag flag;

    if (trg_cell.marked_split_) {
      flag.Add(IntrFlag::SplitL);
    }
    if (src_cell.marked_split_) {
      flag.Add(IntrFlag::SplitR);
    }
    if (src_cell.marked_nb_) {
      flag.Add(IntrFlag::ReadNbR);
    }
    if (trg_cell.marked_nb_) {
      flag.Add(IntrFlag::ReadNbL);
    }
    if (src_cell.marked_attr_) {
      flag.Add(IntrFlag::ReadAttrR);
    }
    if (trg_cell.marked_attr_) {
      flag.Add(IntrFlag::ReadAttrL);
    }

    return flag;
  }

  // Check if cells are split or not in 2-parameter Map
  template<class UserFunct, class...Args>
  static IntrFlag PredSplit2(KeyType trg_key, KeyType src_key, const Data &data, UserFunct f, Args...args) {
    ProxyCell trg_cell(data, nullptr, trg_key);
    ProxyCell src_cell(data, nullptr, src_key);
    
    return PredSplit2(trg_cell, src_cell, f, args...);
  }

  inline int depth() const {
    return this->Base::depth();
  }

  template<class DistanceType>
  inline VecT dX(const ProxyCell &rhs, DistanceType t) const {
    return this->Base::dX((Base&)rhs, t);
  }

  /**
   * \brief Distance Function
   */
  template<class DistanceType>
  inline VecT dX(const Body &b, DistanceType t) const {
    return this->Base::dX(b, t);
  }

  /**
   * \brief Distance Function
   */
  inline VecT dX(const RealBody &b, tapas::CenterClass) const {
    //std::ccout << __FILE__ << ":" << __LINE__ << std::endl;
    //VecT body_pos = ParticlePosOffset<Dim, FP, TSP::kBodyCoordOffset>::vec(&b);
    //return this->Base::dX(body_pos, tapas::Center);
    return this->Base::dX(b, tapas::Center);
  }

  /**
   * \fn FP ProxyCell::width(FP d) const
   */
  inline FP width(FP d) const {
    return this->Base::width(d);
  }

  /**
   * \fn FP ProxyCell::width() const
   */
  inline VecT width() const {
    return this->Base::width();
  }

  /**
   * \fn bool ProxyCell::IsLeaf() const
   */
  inline bool IsLeaf() const {
    return this->Base::IsLeaf();
  }
  
  inline index_t nb() const {
    TAPAS_ASSERT(IsLeaf() && "Cell::nb() is not allowed for non-leaf cells.");
    MarkNb();
    return this->Base::nb();
  }

  inline SubCellIterator<ProxyCell> subcells() {
    Split();
    return SubCellIterator<ProxyCell>(*this);
  }

  inline SubCellIterator<ProxyCell> subcells() const {
    Split();
    return SubCellIterator<ProxyCell>(const_cast<ProxyCell&>(*this));
  }

  inline ProxyCell &subcell(int nch) {
    if (IsLeaf()) {
      std::cerr << "Tapas ERROR: Cell::subcell(int) is called for a leaf cell (in inspector)" << std::endl;
      abort();
    }
      
    TAPAS_ASSERT((index_t)nch < nsubcells());
    Split();

    if (children_.size() == 0) {
      size_t ns = nsubcells();
      children_.resize(ns, nullptr);
      for (size_t i = 0; i < ns; i++) {
        children_[i] = new ProxyCell(data_, clock_, this->Base::Child(i));
      }
    }
    return *children_[nch];
  }

  inline size_t nsubcells() const {
    Split();
    return IsLeaf() ? 0 : (1 << TSP::Dim);
  }

  /**
   * \fn ProxyCell::attr
   */
  const CellAttr &attr() const {
    ReadAttr();
    return attr_;
  }

  /**
   * \fn ProxyCell::bodies()
   */
  ProxyBodyIterator<ProxyCell> bodies() {
    MarkNb();
    return ProxyBodyIterator<ProxyCell>(this);
  }
    
  ProxyBodyIterator<ProxyCell> bodies() const {
    MarkNb();
    return ProxyBodyIterator<ProxyCell>(const_cast<ProxyCell*>(this));
  }

  const Body &body(index_t idx) const { // returns ProxyBody
    // This function is inhibited.
    // Users should use Map()/Reduce() and Cell::bodies() instead.

    TAPAS_ASSERT(IsLeaf() && "Cell::body() is not allowed for a non-leaf cell.");
    TAPAS_ASSERT(idx < nb() && "Body index out of bound. Check nb()." );

    return this->Base::body(idx);
  }
  
  BodyAttr &body_attr(index_t idx) {
    // This function is inhibited.
    // Users should use Map()/Reduce() and Cell::bodies() instead.
    std::cerr << "Cell::body() and Cell::body_attr() is inhibited. Please use Map() API instead." << std::endl;
    exit(-1);
    
    TAPAS_ASSERT(IsLeaf() && "ProxyCell::body_attr() can be called only for leaf cells");
    TAPAS_ASSERT(idx < nb());
    
    return this->Base::body_attr(idx);
  }

  const Data &data() const { return data_; }

  static int IncIfNotNull(int *p) {
    if (p == nullptr) return 1;
    else {
      int v = *p;
      *p += 1;
      return v;
    }
  }

  //protected:
  void ReadAttr() const {
    marked_attr_ = IncIfNotNull(clock_);
  }
  void Split() const {
    marked_split_ = IncIfNotNull(clock_);
  }
  void MarkNb() const {
    marked_nb_ = IncIfNotNull(clock_);
  }
  void MarkModified() {
    marked_modified_ = IncIfNotNull(clock_);
    //std::cout << "ProxyCell::MarkModified() key=" << key() << ", marked_modified_ = " << marked_modified_ << std::endl;
  }

  int IsMarkedModified() const {
    return marked_modified_;
  }

 private:
  
  const Data &data_;

  mutable int marked_attr_;  // attribute is read
  mutable int marked_split_; // split
  mutable int marked_nb_;    // member function nb() is called
  int marked_modified_;
  mutable int *clock_;

  bool is_local_;

  CellAttr attr_;
    
  std::vector<ProxyCell*> children_;
  Mapper mapper_; // FIXME: create Mapper for every ProxyCell is not efficient.
}; // end of class ProxyCell

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_PROXY_CELL_H_
