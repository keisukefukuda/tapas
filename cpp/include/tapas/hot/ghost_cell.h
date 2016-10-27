/**
 * \file Implements Ghost cell class which is used for the Inspector.
 */

#ifndef TAPAS_HOT_GHOST_CELL_H_
#define TAPAS_HOT_GHOST_CELL_H_

#include "tapas/stdcbug.h"
#include "tapas/geometry.h"

namespace tapas {
namespace hot {


#if 0
template<class TSP> class ProxyBodyIterator;

/**
 * @brief A dummy class of Mapper
 */
template<class TSP>
struct GhostMapper {
  GhostMapper() { }
  
  // body
  template<class Funct, class...Args>
  inline void Map(Funct, ProxyBodyIterator<TSP> &, Args...) {
    // empty
  }

  // body x body
  template<class Funct, class ...Args>
  inline void Map(Funct, ProxyBodyIterator<TSP>, ProxyBodyIterator<TSP>, Args...) {
    // empty
  }

  // body iter x body
  template<class Funct, class ...Args>
  inline void Map(Funct, ProxyBodyIterator<TSP>, ProxyBody<TSP> &, Args...) {
    // empty
  }

  // cell x cell
  template<class Funct, class ...Args>
  inline void Map(Funct, ProxyCell<TSP> &, ProxyCell<TSP> &, Args...) {
    // empty
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<ProxyCell<TSP>> &, CellIterator<ProxyCell<TSP>> &, Args...) {
    // empty
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, CellIterator<ProxyCell<TSP>> &, SubCellIterator<ProxyCell<TSP>> &, Args...) {
    // empty
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<ProxyCell<TSP>> &, SubCellIterator<ProxyCell<TSP>> &, Args...) {
    // empty
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<ProxyCell<TSP>> &, Args...) {
    // empty
  }

  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void MapP2(Funct /* f */, ProductIterator<T1_Iter, T2_Iter> /*prod*/, Args.../*args*/) {
    // empty
  }
  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class... Args>
  inline void MapP1(Funct /*f*/, ProductIterator<T1_Iter> /*prod*/, Args.../*args*/) {
    // empty
  }
};

#endif


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
  //using Mapper = MAPPER;
  using Region = REGION_T;
  static const constexpr int Dim = Region::Dim;
  using FP = typename Region::FP;
  using Vec = tapas::Vec<Dim, FP>;

  /**
   * \brief ctor. 
   * \param [in] reg Region of target/source (not the root of whole space)
   * \param [width] width of the ghost cell
   */
  GhostCell(/*Mapper &mapper, */ const Region &reg, const Vec &width)
      : region_(reg)
      , width_(width)
  {
    for (int d = 0; d < Dim; d++) {
      if ((double)reg.width(d) + 1e-9 < width[d]) {
        std::cerr << "d = " << d << std::endl;
        std::cerr << "reg.width(d) = " << reg.width(d) << std::endl;
        std::cerr << "width[d]     = " << width[d] << std::endl;
        std::cerr << "reg.width    = " << reg.width() << std::endl;
        std::cerr << "width        = " << width << std::endl;
      }
      TAPAS_ASSERT((double)reg.width(d) + 1e-9 >= width[d]);
    }
  }

  inline Vec center() const { return region_.center(); }

  template<class CellType>
  inline FP Distance(const CellType &rhs,  tapas::CenterClass) const {
    return sqrt(Distance2(rhs, tapas::CenterClass()));
  }

  // Calculate the center-center distance between the two given ghost cells
  template<class CellType>
  inline FP Distance2(const CellType &rhs,  tapas::CenterClass) const {
    const Region &r1 = region_, &r2 = rhs.region();
    Vec dist = {0.0};
    
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
        dist[dim] = 0;
      } else {
        dist[dim] = std::min(fabs(a-d), fabs(c-b));
      }
      //std::cout << "Dim " << dim << " dist = " << dist[dim] << std::endl;
      //std::cout << std::endl;
    }
    return dist.norm();
  }

  bool operator==(const GhostCell &rhs) const {
    return false;
  }

  const Region& region() const { return region_; }
  Region& region() { return region_; }
  const Vec& width() const { return width_; }
  Vec& width() { return width_; }
  FP width(int d)const { return width_[d]; }

  //Mapper &mapper() { return mapper_; }

 private:
  //Mapper mapper_;
  const Region region_;
  const Vec width_;
};

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_GHOST_CELL_H_
