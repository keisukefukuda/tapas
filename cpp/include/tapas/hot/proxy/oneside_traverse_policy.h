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
  using RealBody = typename Data::BodyType;
  using RealBodyAttr = typename Data::BodyAttrType;

  using Body = ProxyBody<RealBody, RealBodyAttr>;
  using BodyAttr = ProxyBodyAttr<RealBody, RealBodyAttr>;
  
  OnesideTraversePolicy(const Data &data, Reg region, VecT width, int depth)
      : region_(region)
      , width_(width)
      , data_(data)
      , is_leaf_(false)
      , depth_(depth)
      , bodies_()
      , body_attrs_()
  {
    Init();
  }
  
  OnesideTraversePolicy(const Data& data, const OnesideTraversePolicy& rhs)
      : region_(rhs.region_)
      , width_(rhs.width_)
      , data_(data)
      , is_leaf_(false)
      , depth_(rhs.depth_)
      , bodies_()
      , body_attrs_()
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
      , bodies_()
      , body_attrs_()
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

  // for debugging purpose
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

  inline void SetIsLeaf(bool b) {
    is_leaf_ = b;
  }

  // Cell-Cell, Center
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

  // Cell-Cell, Shortest
  inline VecT dX(const OnesideTraversePolicy &rhs, tapas::ShortestClass) const {
    VecT dx = {0.0};
    for (int d = 0; d < Dim; d++) {
      FP a_min = region_.min(d), a_max = region_.max(d);
      FP b_min = rhs.region_.min(d), b_max = rhs.region_.max(d);
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

  // Cell-Body, Center
  inline VecT dX(const VecT& body_pos, tapas::CenterClass) const {
    // todo
    VecT dx = {0.0};
    
    for (int d = 0; d < Dim; d++) {
      if (body_pos[d] < region_.min(d)) {
        dx[d] = region_.min(d) - body_pos[d] + width_[d]/2;
      } else if (body_pos[d] > region_.max(d)) {
        dx[d] = body_pos[d] - region_.max(d) + width_[d]/2;
      } else {
        dx[d] = 0;
      }
    }
    return dx;
  }

  // Cell-Body, Shortest
  inline VecT dX(const VecT& body_pos, tapas::ShortestClass) const {
    // todo
    VecT dx = {0.0};
    
    for (int d = 0; d < Dim; d++) {
      if (body_pos[d] < region_.min(d)) {
        dx[d] = region_.min(d) - body_pos[d];
      } else if (body_pos[d] > region_.max(d)) {
        dx[d] = body_pos[d] - region_.max(d);
      } else {
        dx[d] = 0;
      }
    }
    return dx;
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
    // always returns 1 in the current design
    return 1;
  }

  void InitBodies() {
    if (nb() > 0) {
      auto num_bodies = nb();
      if (bodies_.size() != num_bodies) {
        bodies_.resize(num_bodies);
        body_attrs_.resize(num_bodies);
        memset((void*)bodies_.data(), 0, sizeof(bodies_[0]) * bodies_.size());
        memset((void*)body_attrs_.data(), 0, sizeof(body_attrs_[0]) * body_attrs_.size());
      }
    }
  }

  const Body &body(index_t idx) {
    if (bodies_.size() != nb()) {
      InitBodies();
    }
    
    TAPAS_ASSERT(idx < (index_t)bodies_.size());
    return bodies_[idx];
  }

  BodyAttr &body_attr(index_t idx) {
    if (body_attrs_.size() != nb()) {
      InitBodies();
    }
    return body_attrs_[idx];
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
  std::vector<Body> bodies_; // a vector of proxy body
  std::vector<BodyAttr> body_attrs_; // a vector of proxy body attr
};

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_ONESIDE_TRAVERSE_POLICY_H_

