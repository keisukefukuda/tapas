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

template<class TSP> struct ProxyMapper;

/**
 * ProxyCell
 */
template<class TSP>
class ProxyCell {
  friend ProxyAttr<TSP>;

 public:
  static const constexpr bool Inspector = true;
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  
  using CellType = Cell<TSP>;
  using CellAttr = ProxyAttr<TSP>;
  using RealCellType = CellType;
  
  using BodyAttrType = ProxyBodyAttr<TSP>;
  using BodyType = ProxyBody<TSP>;
  using KeyType = typename tapas::hot::Cell<TSP>::KeyType;
  using SFC = typename tapas::hot::Cell<TSP>::SFC;
  using Data = typename CellType::Data;
  using Reg = Region<Dim, FP>;

  using Threading = typename CellType::Threading;

  using Mapper = ProxyMapper<TSP>;
  using Vec = typename RealCellType::Vec;

  // ctor
  ProxyCell(KeyType key, const Data &data, int *clock = nullptr)
      : key_(key), data_(data)
      , marked_touched_(false), marked_split_(false), marked_body_(false), marked_modified_(false), clock_(clock)
      , is_local_(false), cell_(nullptr), bodies_(), body_attrs_(), attr_(this), parent_(nullptr), children_()
  {
    if (data.ht_.count(key_) > 0) {
      is_local_ = true;
      cell_ = data.ht_.at(key_);
      attr_ = ProxyAttr<TSP>(this, cell_->attr());
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

  // for debug
  inline Reg region() const {
    return RealCellType::CalcRegion(key_, data_.region_);
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
    return key_ == rhs.key_;
  }

  bool operator<(const ProxyCell &rhs) const {
    return key_ < rhs.key_;
  }

  // Check if cells are split or not in 2-parameter Map
  template<class UserFunct, class...Args>
  static SplitType PredSplit2(KeyType trg_key, KeyType src_key, const Data &data, UserFunct f, Args...args) {
    ProxyCell trg_cell(trg_key, data);
    ProxyCell src_cell(src_key, data);

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

  // TODO
  unsigned size() const {
    Touched();
    return 0;
  } // BasicCell::size() in cell.h  (always returns 1)

  /**
   * \fn Vec ProxyCell::center()
   */
  inline Vec center() const {
    Touched();
    return Cell<TSP>::CalcCenter(key_, data_.region_);
  }

  inline int depth() const {
    return SFC::GetDepth(key_);
  }

  /**
   * \brief Distance Function
   */
  inline FP Distance(const ProxyCell &rhs, tapas::CenterClass) const {
    return tapas::Distance<Dim, tapas::CenterClass, FP>::Calc(*this, rhs);
  }

  //inline FP Distance(Cell &rhs, tapas::Edge) {
  //  return tapas::Distance<tapas::Edge, FP>::Calc(*this, rhs);
  //}

  /**
   * \fn FP ProxyCell::width(FP d) const
   */
  inline FP width(FP d) const {
    Touched();
    return Cell<TSP>::CalcRegion(key_, data_.region_).width(d);
  }

  inline bool IsLeaf_real() const {
    if (is_local_) return cell_->IsLeaf();
    else           return data_.max_depth_ <= SFC::GetDepth(key_);
  }

  /**
   * \fn bool ProxyCell::IsLeaf() const
   */
  inline bool IsLeaf() const {
    Touched();
    return IsLeaf_real();
  }

  inline bool IsLocal() const {
    return is_local_;
    //return data_.ht_.count(key_) > 0;
  }

  inline bool IsRoot() const {
    return key_ == 0;
  }

  inline index_t local_nb() {
    return cell_ ? cell_->local_nb() : 0;
  }

  inline index_t nb() {
    Touched();
    if (is_local_) {
      return cell_->nb();
    } else {
      TAPAS_ASSERT(IsLeaf() && "Cell::nb() is not allowed for non-leaf cells.");
      Body();
      return 0;
    }
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
    if (IsLeaf_real()) {
      std::cerr << "Tapas ERROR: Cell::subcell(int) is called for a leaf cell (in inspector)" << std::endl;
      abort();
    }
      
    TAPAS_ASSERT((index_t)nch < nsubcells());
    Split();

    if (children_.size() == 0) {
      size_t ns = nsubcells();
      children_.resize(ns, nullptr);
      for (size_t i = 0; i < ns; i++) {
        children_[i] = new ProxyCell(SFC::Child(key_, i), data_, clock_);
      }
    }
    return *children_[nch];
  }

  inline ProxyCell &parent() {
    TAPAS_ASSERT(key_ != 0);
    if (parent_ == nullptr) {
      parent_ = new ProxyCell(SFC::Parent(key_), data_);
    }

    return *parent_;
  }
    
  inline size_t nsubcells() const {
    Split();
    return IsLeaf_real() ? 0 : (1 << TSP::Dim);
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
  ProxyBodyIterator<TSP> bodies() {
    Touched();
    return ProxyBodyIterator<TSP>(this);
  }
    
  ProxyBodyIterator<TSP> bodies() const {
    Touched();
    return ProxyBodyIterator<TSP>(const_cast<ProxyCell*>(this));
  }

  const ProxyBody<TSP> &body(index_t idx) {
    Touched();
    if (is_local_) {
      TAPAS_ASSERT(IsLeaf() && "Cell::body() is not allowed for a non-leaf cell.");
      TAPAS_ASSERT(idx < cell_->nb() && "Body index out of bound. Check nb()." );

      if (bodies_.size() != cell_->nb()) {
        InitBodies();
      }
    } else {
      // never reach here because remote ProxyCell::nb() always returns 0 in LET mode.
      TAPAS_ASSERT(!"Tapas internal eror: ProxyCell::body_attr() must not be called in LET mode.");
    }

    TAPAS_ASSERT(idx < (index_t)bodies_.size());
    return *bodies_[idx];
  }

  ProxyBodyAttr<TSP> &body_attr(index_t idx) {
    Touched();
    if (is_local_) {
      TAPAS_ASSERT(IsLeaf() && "ProxyCell::body_attr() can be called only for leaf cells");
      TAPAS_ASSERT(idx < cell_->nb());

      if (body_attrs_.size() != cell_->nb()) {
        InitBodies();
      }
    } else {
      // never reach here because remote ProxyCell::nb() always returns 0 in LET mode.
      TAPAS_ASSERT(!"Tapas internal eror: ProxyCell::body_attr() must not be called in LET mode.");
    }
    return *body_attrs_[idx];
  }

  KeyType key() const { return key_; }
  const Data &data() const { return data_; }

  CellType *RealCell() {
    return cell_;
  }

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
  void Body() const {
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
    if (cell_ != nullptr && cell_->nb() > 0) {
      auto num_bodies = cell_->nb();
      if (bodies_.size() != num_bodies) {
        bodies_.resize(num_bodies);
        body_attrs_.resize(num_bodies);
        for (index_t i = 0; i < num_bodies; i++) {
          bodies_[i] = reinterpret_cast<ProxyBody<TSP>*>(&cell_->body(i));
          body_attrs_[i] = reinterpret_cast<ProxyBodyAttr<TSP>*>(&cell_->body_attr(i));
        }
      }
    }
  }

 private:
  KeyType key_;
  const Data &data_;

  mutable int marked_touched_;
  mutable int marked_split_;
  mutable int marked_body_;
  int marked_modified_;
  mutable int *clock_;

  bool is_local_;

  CellType *cell_;
  std::vector<ProxyBody<TSP>*> bodies_;
  std::vector<ProxyBodyAttr<TSP>*> body_attrs_;
  CellAttr attr_;
    
  ProxyCell *parent_;
  std::vector<ProxyCell*> children_;
  Mapper mapper_; // FIXME: create Mapper for every ProxyCell is not efficient.
}; // end of class ProxyCell

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_PROXY_CELL_H_
