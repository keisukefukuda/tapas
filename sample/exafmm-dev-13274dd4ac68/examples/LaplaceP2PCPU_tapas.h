#include "tapas/debug_util.h"
#include "tapas_exafmm.h"

const real_t EPS2 = 0.0;                                        //!< Softening parameter (squared)

extern uint64_t numP2P;

#if !defined(__CUDACC__) && defined(COUNT)
# define INC_P2P do { numP2P++; } while(0)
#else
# define INC_P2P
#endif

#ifdef USE_SCOREP
# include <scorep/SCOREP_User.h>
#else
#define SCOREP_USER_REGION(_1, _2)
#endif

/**
 * Non-mutual version of P2P
 */
struct P2P {
  template<typename _Body, typename _BodyAttr>
  TAPAS_KERNEL
  void operator()(_Body &Bi, _BodyAttr &Bi_attr, const _Body &Bj, const _BodyAttr &, vec3 Xperiodic) {
    INC_P2P;
    SCOREP_USER_REGION("P2P", SCOREP_USER_REGION_TYPE_FUNCTION);

    vec3 dX = Bi.X - Bj.X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;

    if (R2 != 0) {
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi.SRC * Bj.SRC * sqrt(invR2);
      dX *= invR2 * invR;
      TapasFMM::Reduce(Bi, Bi_attr[0], invR);
      TapasFMM::Reduce(Bi, Bi_attr[1], -dX[0]);
      TapasFMM::Reduce(Bi, Bi_attr[2], -dX[1]);
      TapasFMM::Reduce(Bi, Bi_attr[3], -dX[2]);

      //printf("R2 %.10f invR2 %.10f dX %.10f %.10f %.10f\n", R2, invR2, dX[0], dX[1], dX[2]);
    }
  }
};

/**
 * Mutual P2P
 */
struct P2P_mutual {
  template<typename _Body, typename _BodyAttr>
  TAPAS_KERNEL
  void operator()(_Body &Bi, _BodyAttr &Bi_attr, _Body &Bj, _BodyAttr &Bj_attr, vec3 Xperiodic) {
    INC_P2P;
    SCOREP_USER_REGION("P2P", SCOREP_USER_REGION_TYPE_FUNCTION);

    vec3 dX = Bi.X - Bj.X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;

    if (R2 != 0) {
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi.SRC * Bj.SRC * sqrt(invR2);
      dX *= invR2 * invR;
      TapasFMM::Reduce(Bi, Bi_attr[0], invR);
      TapasFMM::Reduce(Bi, Bi_attr[1], -dX[0]);
      TapasFMM::Reduce(Bi, Bi_attr[2], -dX[1]);
      TapasFMM::Reduce(Bi, Bi_attr[3], -dX[2]);

      //printf("R2 %.10f invR2 %.10f dX %.10f %.10f %.10f\n", R2, invR2, dX[0], dX[1], dX[2]);

      if (Bi.X != Bj.X) {
        Bj_attr[0] += invR;
        Bj_attr[1] += dX[0];
        Bj_attr[2] += dX[1];
        Bj_attr[3] += dX[2];
      }
    }
  }
};
