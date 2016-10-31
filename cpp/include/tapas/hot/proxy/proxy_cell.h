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
 * ProxyCell
 */
template<class _TSP, class _POLICY>
class ProxyCell : public _POLICY {
  friend ProxyAttr<ProxyCell>;

  using Base = _POLICY;

 public:
  using TSP = _TSP;
  static const constexpr bool Inspector = true;
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  
  using CellType = Cell<TSP>;
  using CellAttr = ProxyAttr<ProxyCell>;
  using Policy = _POLICY;
  
  using BodyAttrType = ProxyBodyAttr<ProxyCell>;
  using BodyType = ProxyBody<ProxyCell>;
  using KeyType = typename tapas::hot::Cell<TSP>::KeyType;
  using SFC = typename tapas::hot::Cell<TSP>::SFC;
  using Data = typename CellType::Data;
  using Reg = Region<Dim, FP>;

  using Threading = typename CellType::Threading;

  using Mapper = ProxyMapper<ProxyCell>;
  using Vec = typename CellType::Vec;

  // ctor
  template<class...Args>
  ProxyCell(const Data &data, int *clock, Args...args)
      : Base(data, args...), data_(data)
      , marked_touched_(false), marked_split_(false), marked_body_(false), marked_modified_(false), clock_(clock)
      , is_local_(false), bodies_(), body_attrs_(), attr_(this), children_()
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

  ProxyCell(const ProxyCell &rhs) = delete;

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

  // Check if cells are split or not in 2-parameter Map
  template<class UserFunct, class...Args>
  static SplitType PredSplit2(KeyType trg_key, KeyType src_key, const Data &data, UserFunct f, Args...args) {
    ProxyCell trg_cell(data, nullptr, trg_key);
    ProxyCell src_cell(data, nullptr, src_key);

    f(trg_cell, src_cell, args...);

    if (trg_cell.marked_split_ && src_cell.marked_split_) {
      return SplitType::SplitBoth;
    } else if (trg_cell.marked_split_) {
      return SplitType::SplitLeft;
    } else if (src_cell.marked_split_) {
      return SplitType::SplitRight;
    } else if (src_cell.marked_body_) {
      return SplitType::Body;
    } else if (!src_cell.marked_touched_) {
      return SplitType::None;
    } else {
      return SplitType::Approx;
    }
  }

  /**
   * \fn Vec ProxyCell::center()
   */
  inline Vec center() const {
    Touched();
    return this->Base::center();
  }

  inline int depth() const {
    return this->Base::depth();
  }

  /**
   * \brief Distance Function
   */
  inline FP Distance(const ProxyCell &rhs, tapas::CenterClass) const {
    return this->Base::Distance((Base&)rhs, tapas::CenterClass());
  }

  /**
   * \fn FP ProxyCell::width(FP d) const
   */
  inline FP width(FP d) const {
    Touched();
    return this->Base::width(d);
  }

  /**
   * \fn bool ProxyCell::IsLeaf() const
   */
  inline bool IsLeaf() const {
    Touched();
    return this->Base::IsLeaf();
  }

  inline index_t nb() {
    TAPAS_ASSERT(IsLeaf() && "Cell::nb() is not allowed for non-leaf cells.");
    Touched();
    MarkBody();
    
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
    Touched();
    return attr_;
  }

  /**
   * \fn ProxyCell::bodies()
   */
  ProxyBodyIterator<ProxyCell> bodies() {
    Touched();
    return ProxyBodyIterator<ProxyCell>(this);
  }
    
  ProxyBodyIterator<ProxyCell> bodies() const {
    Touched();
    return ProxyBodyIterator<ProxyCell>(const_cast<ProxyCell*>(this));
  }

  const ProxyBody<ProxyCell> &body(index_t idx) {
    Touched();

    TAPAS_ASSERT(IsLeaf() && "Cell::body() is not allowed for a non-leaf cell.");
    TAPAS_ASSERT(idx < nb() && "Body index out of bound. Check nb()." );

    if (bodies_.size() != nb()) {
      InitBodies();
    }

    TAPAS_ASSERT(idx < (index_t)bodies_.size());
    return *bodies_[idx];
  }

  ProxyBodyAttr<ProxyCell> &body_attr(index_t idx) {
    Touched();
    
    TAPAS_ASSERT(IsLeaf() && "ProxyCell::body_attr() can be called only for leaf cells");
    TAPAS_ASSERT(idx < nb());
    
    if (body_attrs_.size() != nb()) {
      InitBodies();
    }
    return *body_attrs_[idx];
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
  void Touched() const {
    marked_touched_ = IncIfNotNull(clock_);
  }
  void Split() const {
    marked_split_ = IncIfNotNull(clock_);
  }
  void MarkBody() const {
    marked_body_ = IncIfNotNull(clock_);
  }
  void MarkModified() {
    marked_modified_ = IncIfNotNull(clock_);
    //std::cout << "ProxyCell::MarkModified() key=" << key() << ", marked_modified_ = " << marked_modified_ << std::endl;
  }

  int IsMarkedModified() const {
    return marked_modified_;
  }

  void InitBodies() {
    auto *cell = Base::RealCell();
    if (cell != nullptr && nb() > 0) {
      auto num_bodies = nb();
      if (bodies_.size() != num_bodies) {
        bodies_.resize(num_bodies);
        body_attrs_.resize(num_bodies);
        for (index_t i = 0; i < num_bodies; i++) {
          bodies_[i] = reinterpret_cast<ProxyBody<ProxyCell>*>(&cell->body(i));
          body_attrs_[i] = reinterpret_cast<ProxyBodyAttr<ProxyCell>*>(&cell->body_attr(i));
        }
      }
    }
  }

 private:
  const Data &data_;

  mutable int marked_touched_;
  mutable int marked_split_;
  mutable int marked_body_;
  int marked_modified_;
  mutable int *clock_;

  bool is_local_;

  std::vector<ProxyBody<ProxyCell>*> bodies_;
  std::vector<ProxyBodyAttr<ProxyCell>*> body_attrs_;
  CellAttr attr_;
    
  std::vector<ProxyCell*> children_;
  Mapper mapper_; // FIXME: create Mapper for every ProxyCell is not efficient.
}; // end of class ProxyCell

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_PROXY_CELL_H_
