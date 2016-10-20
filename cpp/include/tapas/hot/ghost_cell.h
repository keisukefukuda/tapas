/**
 * \file Implements Ghost cell class which is used for the Inspector.
 */

#ifndef TAPAS_HOT_GHOST_CELL_H_
#define TAPAS_HOT_GHOST_CELL_H_

#include "tapas/stdcbug.h"
#include "tapas/geometry.h"

namespace tapas {
namespace hot {

/**
 * \brief Ghost cell class, which is used by the inspector of Tapas.
 *
 * The inspector needs to calculate the possible shortest distance between the two regions
 * (typically target and source regions).
 *
 */
template<class REGION_T>
class GhostCell {
 public:
  using Region = REGION_T;
  static const constexpr int Dim = Region::Dim;
  using FP = typename Region::FP;
  using Vec = tapas::Vec<Dim, FP>;

  /**
   * \brief ctor. 
   * \param [in] reg Region of target/source (not the root of whole space)
   * \param [width] width of the ghost cell
   */
  GhostCell(const Region &reg, FP width)
      : region_(reg), width_(width) {
    for (int d = 0; d < Dim; d++) {
      TAPAS_ASSERT(reg.width(d) >= width);
    }
  }

  inline Vec center() const { return region_.center(); }

  inline FP Distance(const GhostCell &rhs,  tapas::CenterClass) const {
    return sqrt(Distance2(rhs, tapas::CenterClass()));
  }

  // Calculate the center-center distance between the two given ghost cells
  inline FP Distance2(const GhostCell &rhs,  tapas::CenterClass) const {
    const Region &r1 = region_, &r2 = rhs.region_;
    Vec dist = {0.0};
    
    for (int dim = 0; dim < Dim; dim++) {
      // range of movement of the center points.
      FP w1 = width_;
      FP w2 = rhs.width_;
      FP a = r1.max(dim) - w1/2, b = r1.min(dim) + w1/2;
      FP c = r2.max(dim) - w2/2, d = r2.min(dim) + w2/2;

      TAPAS_ASSERT(a >= b);
      TAPAS_ASSERT(c >= d);

      // if the two ranges overlap, then the shortest distance of r1 and r2 is 0.
      bool overlp = false; // overlapped

      if (a <= c && d <= b) overlp = true; // r2 is included in r1
      if (c <= a && b <= d) overlp = true; // r1 is included in r2
      if (a <= c && c <= b) overlp = true;
      if (a <= d && d <= b) overlp = true;

      if (overlp) {
        dist[dim] = 0;
      } else {
        dist[dim] = std::min(fabs(a-d), fabs(c-b));
      }
    }
    return dist.norm();
  }
  
 private:
  const Region region_;
  const double width_;
};

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_GHOST_CELL_H_
