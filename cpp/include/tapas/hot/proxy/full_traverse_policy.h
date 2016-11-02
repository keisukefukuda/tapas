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
  
  using BodyAttrType = ProxyBodyAttr<TSP>;
  using BodyType = ProxyBody<TSP>;
  using KeyType = typename tapas::hot::Cell<TSP>::KeyType;
  using SFC = typename tapas::hot::Cell<TSP>::SFC;
  using Data = typename CellType::Data;
  using Reg = Region<Dim, FP>;
  using Vec = typename CellType::Vec;

 public:
  FullTraversePolicy(const Data &data, KeyType key)
      : key_(key)
      , data_(data)
      , is_leaf_(false)
      , is_local_(false)
      , real_cell_(nullptr)
  {
    Init();
  }
  
  FullTraversePolicy(const Data& data, const FullTraversePolicy& rhs)
      : key_(rhs.key_)
      , data_(rhs.data_)
      , is_leaf_(false)
      , is_local_(false)
      , real_cell_(nullptr)
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

  inline Vec dX(const FullTraversePolicy& rhs, tapas::CenterClass) const {
    return center() - rhs.center();
  }

  inline FP Distance(const Vec& body_pos, tapas::CenterClass) const {
    return dX(body_pos, tapas::CenterClass()).norm();
  }

  inline Vec dX(const Vec& body_pos, tapas::CenterClass) const {
    return center() - body_pos;
  }

  inline FP width(int d) const {
    return Cell<TSP>::CalcRegion(key_, data_.region_).width(d);
  }

  inline bool IsLeaf() const {
    if (is_local_) return real_cell_->IsLeaf();
    else           return data_.max_depth_ <= SFC::GetDepth(key_);
  }

  inline index_t local_nb() {
    return real_cell_ ? real_cell_->local_nb() : 0;
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

 public:
  inline bool IsRoot() const { // being public for debugging
    return key_ == 0;
  }

  KeyType key() const { return key_; } // being public for debugging purpose

  inline Vec center() const {
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
};

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_FULL_TRAVERSE_POLICY_H_

