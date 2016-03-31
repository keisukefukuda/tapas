#ifndef __TAPAS_HOT_GEOMETRY_H__
#define __TAPAS_HOT_GEOMETRY_H__

#include "tapas/vec.h"

namespace tapas {

template<typename VEC>
bool Separated(VEC &xmax, VEC &xmin, VEC &ymax, VEC &ymin) {
  const constexpr int Dim = VEC::Dim;
  
  bool separated = false;

  for (int d = 0; d < Dim; d++) {
    separated |= (xmax[d] <= ymin[d] || ymax[d] <= xmin[d]);
  }

  return separated;
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

template<typename DIST_TYPE, typename FP> // takes DistanceType
struct Distance;

/**
 * \brief Struct to provide center-based distance functions
 */
template<typename FP>
struct Distance<CenterClass, FP> {
  template<typename Cell>
  static inline FP Calc(Cell &c1, Cell &c2) {
    return (c1.center() - c2.center()).norm();
  }

  /**
   * \brief Calculate distance between a target cell and source cell, where
   *        the target cell is a pseudo-cell (or region of the local process)
   */
  template<int DIM>
  static inline FP CalcApprox(const Vec<DIM, FP> &trg_max,
                              const Vec<DIM, FP> &trg_min,
                              const Vec<DIM, FP> &src_max,
                              const Vec<DIM, FP> &src_min) {
    Vec<DIM, FP> trg_ctr, src_ctr;
    Vec<DIM, FP> decision = 0.0;
    
    for (int d = 0; d < DIM; d++) {
      FP Rt = trg_max[d] - trg_min[d];
      FP Rs = src_max[d] - src_min[d];
      FP sctr = src_ctr[d] = (src_max[d] + src_min[d]) / 2;
      
      if (Rs >= Rt) {
        trg_ctr[d] = (trg_max[d] + trg_min[d]) / 2;
        decision[d] = 1.0;
      } else if (sctr < trg_min[d] + Rs/2) {
        trg_ctr[d] = trg_min[d] + Rs/2;
        decision[d] = 2.0;
      } else if (sctr > trg_max[d] - Rs/2) {
        trg_ctr[d] = trg_max[d] - Rs/2;
        decision[d] = 3.0;
      } else {
        trg_ctr[d] = src_ctr[d];
        decision[d] = 4.0;
      }
    }
    
    if (getenv("TAPAS_DEBUG_TMP")) {
      std::cout << "CalcApprox: " << "trg_max =" << trg_max << std::endl;
      std::cout << "CalcApprox: " << "trg_min =" << trg_min << std::endl;
      std::cout << "CalcApprox: " << "Rt      =" << (trg_max - trg_min) << std::endl;
      std::cout << "CalcApprox: " << "src_max =" << src_max << std::endl;
      std::cout << "CalcApprox: " << "src_min =" << src_min << std::endl;
      std::cout << "CalcApprox: " << "Rs      =" << (src_max - src_min) << std::endl;
      std::cout << "CalcApprox: " << "trg_ctr =" << trg_ctr << std::endl;
      std::cout << "CalcApprox: " << "dec     =" << decision << std::endl;
    }
    
    return (src_ctr - trg_ctr).norm();
  }
};

} // namespace tapas

#endif // __TAPAS_HOT_GEOMETRY_H__
