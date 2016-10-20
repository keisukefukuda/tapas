#ifndef TAPAS_GEOMETRY_H_
#define TAPAS_GEOMETRY_H_

#include <iterator>
#include <type_traits>

#include "tapas/vec.h"
#include "tapas/basic_types.h"

namespace tapas {

template<typename VEC>
bool Separated(const VEC &xmax, const VEC &xmin, const VEC &ymax, const VEC &ymin) {
  const constexpr int Dim = VEC::Dim;

  bool separated = false;

 for (int d = 0; d < Dim; d++) {
    separated |= (xmax[d] <= ymin[d] || ymax[d] <= xmin[d]);
  }

  return separated;
}

template<typename Region>
bool Separated(const Region &x, const Region &y) {
  const constexpr int Dim = Region::Dim;

  bool separated = false;

  for (int d = 0; d < Dim; d++) {
    separated |= (x.max(d) <= y.min(d) || y.max(d) <= x.min(d));
  }

  return separated;
}

template<typename Iter, typename Region>
bool Separated(Iter beg, Iter end, Region &y) {
  using value_type = typename std::iterator_traits<Iter>::value_type;
  static_assert(std::is_same<typename std::remove_const<value_type>::type, typename std::remove_const<Region>::type>::value,
                "Inconsistent Types");

  bool r = true;
  for (Iter iter = beg; iter != end; iter++) {
    r = r && Separated(*iter, y);
  }

  return r;
}

// Returns if X includes Y
template<typename VEC>
bool Includes(VEC &xmax, VEC &xmin, VEC &ymax, VEC &ymin) {
  const constexpr int Dim = VEC::Dim;
  bool res = true;

  for (int d = 0; d < Dim; d++) {
    res &= (xmax[d] >= ymax[d] && ymin[d] >= xmin[d]);
  }

  return res;
}

class CenterClass {} Center;
class EdgeClass{} Edge;

template<int Dim, typename DIST_TYPE, typename FP> // takes DistanceType
struct Distance;

/**
 * \brief Struct to provide center-based distance functions
 */
template<int _DIM, typename _FP>
struct Distance<_DIM, CenterClass, _FP> {
  static const constexpr int Dim = _DIM;
  using FP = _FP;
  using Reg = Region<Dim, FP>;

  template<typename Cell>
  static inline FP Calc(const Cell &c1, const Cell &c2) {
    return (c1.center() - c2.center()).norm();
  }
};

} // namespace tapas

#endif // TAPAS_GEOMETRY_H_
