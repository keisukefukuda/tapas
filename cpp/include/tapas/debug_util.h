#ifndef TAPAS_DEBUG_UTIL_H_
#define TAPAS_DEBUG_UTIL_H_

#include "tapas/common.h"
#include "tapas/basic_types.h"

namespace tapas {
namespace debug {

template <int DIM, class FP, class BT, int POS_OFFSET>
void PrintBodies(const BT *b, int nb, std::ostream &os) {
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < DIM; ++j) {    
      FP p = *ParticlePosOffset<DIM, FP, POS_OFFSET>::pos((void *)&(b[i]), j);
      os << p << " ";
    }
    os << std::endl;
  }
}

} // debug
} // tapas

#endif // TAPAS_DEBUG_UTIL_H_
