#ifndef TAPAS_GEOMETRY_H_
#define TAPAS_GEOMETRY_H_

#include <iterator>
#include <type_traits>

#include "tapas/vec.h"
#include "tapas/basic_types.h"

namespace tapas {

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
class ShortestClass{} Shortest;

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
