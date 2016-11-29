#ifndef TAPAS_MAP_H_
#define TAPAS_MAP_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <functional>
#include <type_traits>

#include "tapas/debug_util.h"
#include "tapas/cell.h"
#include "tapas/iterator.h"

namespace {
namespace iter = tapas::iterator;
}

namespace tapas {

// Utility classes to wrap a user's function and arguments
// into a simple function object that takes only cells.
template<int ...SS>
struct Seq { };

template<int N, int...SS>
struct GenSeq : GenSeq<N-1, N-1, SS...> { };

template<int...SS>
struct GenSeq<0, SS...> {
  typedef Seq<SS...> type;
};

template<class Funct, class ...Args>
struct CallbackWrapper {
  Funct f_;
  std::tuple<Args...> args_;

  CallbackWrapper(Funct f, Args... args) : f_(f), args_(args...) {}

  template<class CellType1, class CellType2>
  void operator()(CellType1 &c1, CellType2 &c2) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    dispatch(c1, c2, typename GenSeq<sizeof...(Args)>::type());
  }
  
  template<class CellType1, class CellType2, int...SS>
  INLINE void dispatch(CellType1 &c1, CellType2 &c2, Seq<SS...>) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f_(c1, c2, std::get<SS>(args_)...);
  }
};

} // namespace tapas

#endif // TAPAS_MAP_H_
