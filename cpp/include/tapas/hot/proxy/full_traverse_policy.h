#ifndef TAPAS_HOT_PROXY_FULL_TRAVERSE_POLICY_H_
#define TAPAS_HOT_PROXY_FULL_TRAVERSE_POLICY_H_

namespace tapas {
namespace hot {
template<class TSP> class Cell;

namespace proxy {

template<class TSP>
class FullTraversePolicy {
 protected:
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  
  using CellType = Cell<TSP>;
  using CellAttr = ProxyAttr<TSP>;
  using RealCellType = CellType;
  using RealBody = typename TSP::Body;
  using RealBodyAttr = typename TSP::BodyAttr;
  
  using Body = ProxyBody<RealBody, RealBodyAttr>;
  using BodyAttr = ProxyBodyAttr<RealBody, RealBodyAttr>;
  
  using KeyType = typename tapas::hot::Cell<TSP>::KeyType;
  using SFC = typename tapas::hot::Cell<TSP>::SFC;
  using Data = typename CellType::Data;
  using Reg = Region<Dim, FP>;
  using VecT = typename CellType::VecT;

 public:
  FullTraversePolicy(const Data &data, KeyType key)
      : key_(key)
      , data_(data)
      , is_leaf_(false)
      , is_local_(false)
      , real_cell_(nullptr)
      , bodies_()
      , body_attrs_()
  {
    Init();
  }
  
  FullTraversePolicy(const Data& data, const FullTraversePolicy& rhs)
      : key_(rhs.key_)
      , data_(rhs.data_)
      , is_leaf_(false)
      , is_local_(false)
      , real_cell_(nullptr)
      , bodies_()
      , body_attrs_()
  {
    (void)data;
    TAPAS_ASSERT(&data == &rhs.data_);
    Init();
  }
  
  FullTraversePolicy(const Data& data, FullTraversePolicy&& rhs)
      : key_(rhs.key_)
      , data_(rhs.data_)
      , is_leaf_(false)
      , is_local_(false)
      , real_cell_(nullptr)
      , bodies_()
      , body_attrs_()
  {
    TAPAS_ASSERT(&data == &rhs.data_);
    Init();
  }

  bool operator==(const FullTraversePolicy& rhs) {
    return (&data_ == &rhs.data_) && (key_ == rhs.key_);
  }

 protected:
  void Init() {
    if (data_.ht_.count(key_) > 0) {
      is_local_ = true;
      real_cell_ = data_.ht_.at(key_);
      is_leaf_ = real_cell_->IsLeaf();
    } else {
      is_leaf_ = SFC::GetDepth(key_) >= data_.max_depth_;
    }

    InitBodies();
  }
  
  /**
   * \brief If the proxy cell is backed by a real cell, returns a pointer to it. Otherwise returns nullptr.
   */
  CellType *RealCell() {
    return real_cell_;
  }

  const CellType *RealCell() const {
    return real_cell_;
  }

  inline Reg region() const {
    return CellType::CalcRegion(key_, data_.region_);
  }

  // Cell-Cell, Shortest
  inline VecT dX(const FullTraversePolicy& rhs, tapas::ShortestClass) const {
    Reg reg1 = region();
    Reg reg2 = rhs.region();
    VecT dx = {0.0};
    for (int d = 0; d < Dim; d++) {
      FP a_min = reg1.min(d), a_max = reg1.max(d);
      FP b_min = reg2.min(d), b_max = reg2.max(d);
      if ((b_min <= a_min && a_min <= b_max) || (b_min <= a_max && a_max <= b_max)) {
        dx[d] = 0;
      } else if (a_min <= b_min && b_max <= a_max) {
        dx[d] = 0;
      } else if (b_min <= a_min && a_max <= b_max) {
        dx[d] = 0;
      } else if (a_max < b_min) {
        dx[d] = b_min - a_max;
      } else if (b_max < a_min) {
        dx[d] = a_min - b_max;
      } else {
        assert(0);
      }
    }
    return dx;
  }

  // Cell-Cell, Center
  inline VecT dX(const FullTraversePolicy& rhs, tapas::CenterClass) const {
    return center() - rhs.center();
  }

  // Cell-Body, Shortest
  inline VecT dX(const VecT& body_pos, tapas::ShortestClass) const {
    return center() - body_pos;
  }

  // Cell-Cell, Center
  inline VecT dX(const VecT& body_pos, tapas::CenterClass) const {
    return center() - body_pos;
  }

  inline FP width(int d) const {
    return Cell<TSP>::CalcRegion(key_, data_.region_).width(d);
  }

  inline VecT width() const {
    return Cell<TSP>::CalcRegion(key_, data_.region_).width();
  }

  inline bool IsLeaf() const {
    if (is_local_) return real_cell_->IsLeaf();
    else           return data_.max_depth_ <= SFC::GetDepth(key_);
  }

  inline index_t local_nb() {
    return real_cell_ ? real_cell_->local_nb() : data_.ncrit_;
  }

  inline index_t nb() const {
    if (is_local_) {
      return real_cell_->nb();
    } else {
      return 0;
    }
  }

  FullTraversePolicy Child(int nth) const {
    return FullTraversePolicy(data_, SFC::Child(key_, nth));
  }

  void InitBodies() const {
    auto *cell = RealCell();

    if (cell != nullptr && cell->IsLeaf() && nb() > 0) {
      auto num_bodies = nb();
      if (bodies_.size() != num_bodies) {
        bodies_.resize(num_bodies);
        body_attrs_.resize(num_bodies);
        for (index_t i = 0; i < num_bodies; i++) {
          bodies_[i] = const_cast<Body*>(reinterpret_cast<const Body*>(&cell->body(i)));
          body_attrs_[i] = const_cast<BodyAttr*>(reinterpret_cast<const BodyAttr*>(&cell->body_attr(i)));
        }
      }
    }
  }
  
 public:
  const Body &body(index_t idx) const {
    return *bodies_[idx];
  }
  Body &body(index_t idx) {
    return *bodies_[idx];
  }

  BodyAttr &body_attr(index_t idx) {
    return *body_attrs_[idx];
  }

  const BodyAttr &body_attr(index_t idx) const {
    return *body_attrs_[idx];
  }

  inline bool IsRoot() const { // being public for debugging
    return key_ == 0;
  }

  KeyType key() const { return key_; } // being public for debugging purpose

  inline VecT center() const {
    return Cell<TSP>::CalcCenter(key_, data_.region_);
  }

  inline int depth() const {
    return SFC::GetDepth(key_);
  }

 protected:
  KeyType key_;
  const Data &data_;
  bool is_leaf_;
  bool is_local_;
  CellType *real_cell_;
  mutable std::vector<Body*> bodies_;
  mutable std::vector<BodyAttr*> body_attrs_;
};

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_FULL_TRAVERSE_POLICY_H_

