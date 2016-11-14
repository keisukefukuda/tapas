#ifndef TAPAS_HOT_PROXY_ONESIDE_TRAVERSE_POLICY_H_
#define TAPAS_HOT_PROXY_ONESIDE_TRAVERSE_POLICY_H_

#include "tapas/common.h"
#include "tapas/geometry.h"
#include "tapas/vec.h"
#include "tapas/basic_types.h"

namespace tapas {
namespace hot {
template<class TSP> class Cell;

namespace proxy {

template<int _DIM, class _FP, class Data>
class OnesideTraversePolicy {
 public:
  static const constexpr int Dim = _DIM;
  using FP = _FP;
  using VecT = Vec<Dim, FP>;
  using Reg = tapas::Region<Dim, FP>;
  using CellType = typename Data::CellType;
  
  OnesideTraversePolicy(const Data &data, Reg region, VecT width, int depth)
      : region_(region)
      , width_(width)
      , data_(data)
      , is_leaf_(false)
      , depth_(depth)
  {
    Init();
  }
  
  OnesideTraversePolicy(const Data& data, const OnesideTraversePolicy& rhs)
      : region_(rhs.region_)
      , width_(rhs.width_)
      , data_(data)
      , is_leaf_(false)
      , depth_(rhs.depth_)
  {
    TAPAS_ASSERT(&data == &rhs.data_);
    Init();
  }
  
  OnesideTraversePolicy(const Data& data, OnesideTraversePolicy&& rhs)
      : region_(rhs.region_)
      , width_(rhs.width_)
      , data_(data)
      , is_leaf_(false)
      , depth_(rhs.depth_)
  {
    TAPAS_ASSERT(&data == &rhs.data_);
    Init();
  }

  OnesideTraversePolicy Child(int) const {
    return OnesideTraversePolicy(data_, region_, width_/2, depth_+1);
  }

  bool operator==(const OnesideTraversePolicy& rhs) {
    return (&data_  == &rhs.data_)
        && (width_  == rhs.width_)
        && (region_ == rhs.region_)
        && (depth_  == rhs.depth_);
  }

  typename CellType::KeyType key() const { return 0; }

 protected:
  void Init() {
    is_leaf_ = (depth_ >= data_.max_depth_);

#ifdef TAPAS_DEBUG
    for (int d = 0; d < Dim; d++) {
      if (width_[d] > region_.width(d)) {
        double diff = width_[d] - region_.width(d);
        diff = diff * diff;
        if (diff > 1e-15) {
          std::cerr << "Dim = " << d << std::endl;
          std::cerr << "\twidth = " << width_[d] << std::endl;
          std::cerr << "\tregion = [" << region_.max()[d] << ", " << region_.min()[d] << "]"
                    << " = " << (region_.max()[d] - region_.min()[d])
                    << std::endl;
          std::cerr << "\tdiff = " << std::scientific << diff << std::endl;
        }
        TAPAS_ASSERT(diff < 1e-15);
      }
    }
#endif
  }

  CellType *RealCell() const { return nullptr; }

 public:
  // inline Reg region() const {
    
  // }

  // inline VecT center() const {
  //   return Cell<TSP>::CalcCenter(key_, data_.region_);
  // }

  inline int depth() const {
    return depth_;
  }

  inline VecT dX(const OnesideTraversePolicy &rhs, tapas::CenterClass) const {
    const Reg &r1 = region_, &r2 = rhs.region_;
    VecT dX = {0.0};
    
    for (int dim = 0; dim < Dim; dim++) {
      // range of movement of the center points.
      FP w1 = width_[dim];
      FP w2 = rhs.width(dim);
      FP a = r1.max(dim) - w1/2, b = r1.min(dim) + w1/2;
      FP c = r2.max(dim) - w2/2, d = r2.min(dim) + w2/2;

      // std::cout << "Dim " << dim << " a = " << a << std::endl;
      // std::cout << "Dim " << dim << " b = " << b << std::endl;
      // std::cout << "Dim " << dim << " c = " << c << std::endl;
      // std::cout << "Dim " << dim << " d = " << d << std::endl;

      if (a < b && ((double)a-b)*(a-b) < 1e-12) {
        a = b;
      }
      if (c < d && ((double)c-d)*(c-d) < 1e-12) {
        c = d;
      }

      if (a < b) {
        std::cerr << "dim = " << dim << std::endl;
        std::cerr << "r1 = [" << r1.max()[dim] << " , " << r2.min()[dim] << "]" << std::endl;
        std::cerr << "w1 = " << w1 << std::endl;
        std::cerr << "a = " << a << std::endl;
        std::cerr << "b = " << b << std::endl;
      }

      TAPAS_ASSERT(a >= b);
      TAPAS_ASSERT(c >= d);

      // if the two ranges overlap, then the shortest distance of r1 and r2 is 0.
      bool overlp = false; // overlapped

      if (b <= d && c <= a) overlp = true; // r2 is included in r1
      if (d <= b && a <= c) overlp = true; // r1 is included in r2
      if (b <= c && c <= a) overlp = true;
      if (b <= d && d <= a) overlp = true;

      //std::cout << "Dim " << dim << " overlap = " << overlp << std::endl;

      if (overlp) {
        dX[dim] = 0;
      } else {
        dX[dim] = std::min(fabs(a-d), fabs(c-b));
      }
      //std::cout << "Dim " << dim << " dX = " << dX[dim] << std::endl;
      //std::cout << std::endl;
    }

    return dX;
  }

  inline FP Distance(const OnesideTraversePolicy& rhs, tapas::CenterClass) const {
    return dX(rhs, tapas::CenterClass()).norm();
  }

  inline FP Distance(const VecT& body_pos, tapas::CenterClass) const {
    return dX(body_pos, tapas::CenterClass()).norm();
  }

  inline VecT dX(const VecT& body_pos, tapas::CenterClass) const {
    // todo
    (void) body_pos;
    return 0;
  }

  inline FP width(int d) const {
    return width_[d];
  }

  inline VecT width() const {
    return width_;
  }

  inline bool IsLeaf() const {
    return is_leaf_;
  }

  inline index_t nb() const {
    return 0;
  }

 public:
  inline bool IsRoot() const { // being public for debugging
    return depth_ == 0;
  }

 protected:
  Reg region_;
  VecT width_;
  const Data &data_;
  bool is_leaf_;
  int depth_;
};

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_ONESIDE_TRAVERSE_POLICY_H_

