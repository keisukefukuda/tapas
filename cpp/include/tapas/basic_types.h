#ifndef TAPAS_BASIC_TYPES_H_
#define TAPAS_BASIC_TYPES_H_

#include <cstdint>

#include "tapas/common.h"
#include "tapas/vec.h"

namespace tapas {

template <class FP, class T>
FP REAL(const T &x) {
  return (FP)x;
}

template <int _DIM, typename _FP> // Tapas Static Params (define in common.h)
class Region {
 public:
  static const constexpr int Dim = _DIM;
  using FP = _FP;

 private:
  Vec<Dim, FP> min_;
  Vec<Dim, FP> max_;

 public:
  Region(const Vec<Dim, FP> &coord_min, const Vec<Dim, FP> &coord_max):
      min_(coord_min), max_(coord_max) {
#ifdef TAPAS_DEBUG
    for (int i = 0; i < Dim; i++) {
      if (!(min_[i] <= max_[i])) {
        std::cerr << "min_ = " << min_[i] << ", max_ = " << max_[i] << std::endl;
      }
      TAPAS_ASSERT(min_[i] <= max_[i]);
    }
#endif
  }

  Region() {}

  Region(std::initializer_list<FP> cmin, std::initializer_list<FP> cmax)
      : min_(cmin), max_(cmax) {
#ifdef TAPAS_DEBUG
    for (int i = 0; i < Dim; i++) {
      if (!(min_[i] <= max_[i])) {
        std::cerr << "min_ = " << min_[i] << ", max_ = " << max_[i] << std::endl;
      }
      TAPAS_ASSERT(min_[i] <= max_[i]);
    }
#endif
  }

  inline Vec<Dim, FP> &min() {
    return min_;
  }
  inline FP min(int d) const {
    return min_[d];
  }
  inline FP &min(int d) {
    return min_[d];
  }
  inline const Vec<Dim, FP> &min() const {
    return min_;
  }
  inline Vec<Dim, FP> &max() {
    return max_;
  }
  inline const Vec<Dim, FP> &max() const {
    return max_;
  }
  inline FP max(int d) const {
    return max_[d];
  }
  inline FP &max(int d) {
    return max_[d];
  }
  inline FP width(int d) const {
    TAPAS_ASSERT(max_[d] >= min_[d]);
    return max_[d] - min_[d];
  }
  inline const Vec<Dim, FP> width() const {
    return max_ - min_;
  }
  inline const Vec<Dim, FP> center() const {
    return (max_ + min_) / 2;
  }
  inline const FP center(int d) const {
    return (max_[d] + min_[d]) / 2;
  }

  Region PartitionBSP(int pos) const {
    Region sr = *this;
    for (int i = 0; i < Dim; ++i) {
      FP center = sr.min(i) + sr.width(i) / 2;
      if ((pos & 1) == 0) {
        sr.max(i) = center;
      } else {
        sr.min(i) = center;
      }
      pos >>= 1;
    }
    return sr;
  }

  std::ostream &Print(std::ostream &os) const {
    os << "{" << min_ << ", " << max_ << "}";
    return os;
  }

  static Region<Dim, FP> join(const Region<Dim, FP>& r) { return r; }
  static Region<Dim, FP> join(Region<Dim, FP>&& r) { return r; }
  
  template <class...Args>
  static Region<Dim,FP> join(const Region<Dim, FP>& r1, const Region<Dim, FP>& r2, Args...args) {
    Vec<Dim, FP> cmax, cmin;
    for (int d = 0; d < Dim; d++) {
      cmax[d] = std::max(r1.max(d), r2.max(d));
      cmin[d] = std::min(r1.min(d), r2.min(d));
    }
    return join(Region<Dim,FP>(cmin,cmax), args...);
  }

  template<class Iter>
  static Region<Dim, FP> join(Iter beg, Iter end) {
    Vec<Dim, FP> cmax = beg.max(), cmin = beg.min();
    
    for (auto iter = beg + 1; iter != end; iter++) {
      for (int d = 0; d < Dim; d++) {
        cmax[d] = std::max(cmax[d], iter->max(d));
        cmin[d] = std::min(cmin[d], iter->min(d));
      }
    }
  }
};

template <class TSP>
std::ostream &operator<<(std::ostream &os, const Region<TSP::Dim, typename TSP::FP> &r) {
    return r.Print(os);
}

/**
 * @brief returns a pointer to Dim-length array of particle coordinate.
 * @tparam Dim dimension
 * @tparam FP Floating point type (usually float or double)
 * @tparam Byte offset to coordinate values
 */
template <int Dim, class FP, int OFFSET>
struct ParticlePosOffset {
    static FP *pos(const void *base, int dim) {
        return (FP *)(((intptr_t)base) + OFFSET + dim * sizeof(FP));
    }
    static Vec<Dim, FP> vec(const void *base) {
        FP *p = (FP *)(((intptr_t)base) + OFFSET);
        return Vec<Dim, FP>(p);
    }
};
} // namespace tapas

#endif // TAPAS_BASIC_TYPES_H_
