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
  }
  
  /**
   * \brief If the proxy cell is backed by a real cell, returns a pointer to it. Otherwise returns nullptr.
   */
  CellType *RealCell() {
    return real_cell_;
  }

  inline Reg region() const {
    CellType::CalcRegion(key_, data_.region_);
  }

  inline FP Distance(const FullTraversePolicy& rhs, tapas::CenterClass) const {
    return dX(rhs, tapas::CenterClass()).norm();
  }

  inline VecT dX(const FullTraversePolicy& rhs, tapas::CenterClass) const {
    return center() - rhs.center();
  }

  inline FP Distance(const VecT& body_pos, tapas::CenterClass) const {
    return dX(body_pos, tapas::CenterClass()).norm();
  }

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

  void InitBodies() {
    auto *cell = RealCell();
    if (cell != nullptr && nb() > 0) {
      auto num_bodies = nb();
      if (bodies_.size() != num_bodies) {
        bodies_.resize(num_bodies);
        body_attrs_.resize(num_bodies);
        for (index_t i = 0; i < num_bodies; i++) {
          bodies_[i] = reinterpret_cast<Body*>(&cell->body(i));
          body_attrs_[i] = reinterpret_cast<BodyAttr*>(&cell->body_attr(i));
        }
      }
    }
  }

 public:
  const Body &body(index_t idx) {
    if (bodies_.size() != nb()) {
      InitBodies();
    }
    
    TAPAS_ASSERT(idx < (index_t)bodies_.size());
    return *bodies_[idx];
  }

  BodyAttr &body_attr(index_t idx) {
    if (body_attrs_.size() != nb()) {
      InitBodies();
    }
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
  std::vector<Body*> bodies_;
  std::vector<BodyAttr*> body_attrs_;
};

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_FULL_TRAVERSE_POLICY_H_

