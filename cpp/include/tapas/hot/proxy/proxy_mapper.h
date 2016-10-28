#ifndef TAPAS_HOT_PROXY_PROXY_BODY_MAPPER_H_
#define TAPAS_HOT_PROXY_PROXY_BODY_MAPPER_H_

#include "tapas/hot/proxy/proxy_cell.h"
#include "tapas/hot/proxy/proxy_body.h"

namespace tapas {
namespace hot {
namespace proxy {

template<class PROXY_CELL> class ProxyBodyIterator;

/**
 * @brief A dummy class of Mapper
 */
template<class PROXY_CELL>
struct ProxyMapper {
  ProxyMapper() { }
  
  // body
  template<class Funct, class...Args>
  inline void Map(Funct, ProxyBodyIterator<PROXY_CELL> &, Args...) {
    // empty
  }

  // body x body
  template<class Funct, class ...Args>
  inline void Map(Funct, ProxyBodyIterator<PROXY_CELL>, ProxyBodyIterator<PROXY_CELL>, Args...) {
    // empty
  }

  // body iter x body
  template<class Funct, class ...Args>
  inline void Map(Funct, ProxyBodyIterator<PROXY_CELL>, ProxyBody<PROXY_CELL> &, Args...) {
    // empty
  }

  // cell x cell
  template<class Funct, class ...Args>
  inline void Map(Funct, PROXY_CELL &, PROXY_CELL &, Args...) {
    // empty
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<PROXY_CELL> &, CellIterator<PROXY_CELL> &, Args...) {
    // empty
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, CellIterator<PROXY_CELL> &, SubCellIterator<PROXY_CELL> &, Args...) {
    // empty
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<PROXY_CELL> &, SubCellIterator<PROXY_CELL> &, Args...) {
    // empty
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct, SubCellIterator<PROXY_CELL> &, Args...) {
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


} // proxy
} // hot
} // tapas

#endif // TAPAS_HOT_PROXY_PROXY_BODY_MAPPER_H_
