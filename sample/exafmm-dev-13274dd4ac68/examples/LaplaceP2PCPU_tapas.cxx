#include "tapas_exafmm.h"

const real_t EPS2 = 0.0;                                        //!< Softening parameter (squared)

#ifdef TAPAS_USE_VECTORMAP

struct tapas_kernel::P2P {

  P2P() {}

#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  void operator() (Body* Bi, Body* Bj, kvec4 &biattr, vec3 Xperiodic) {
    vec3 dX = Bi->X - Bj->X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;
    if (R2 != 0) {
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi->SRC * Bj->SRC * sqrt(invR2);
      dX *= invR2 * invR;
      biattr[0] += invR;
      biattr[1] -= dX[0];
      biattr[2] -= dX[1];
      biattr[3] -= dX[2];
    }
  }
};

#else /* TAPAS_USE_VECTORMAP */

void tapas_kernel::P2P(Tapas::BodyIterator &Bi, Tapas::BodyIterator &Bj, vec3 Xperiodic) {
  kreal_t pot = 0; 
  kreal_t ax = 0;
  kreal_t ay = 0;
  kreal_t az = 0;
  vec3 dX = Bi->X - Bj->X - Xperiodic;
  real_t R2 = norm(dX) + EPS2;
  //std::cerr << "tapas_kernel::P2P (" << Bi.index() << "," << Bj.index() << ")" << std::endl;
  if (R2 != 0) {
    auto orig_attr_i = Bi.attr();
    auto orig_attr_j = Bj.attr();
    real_t invR2 = 1.0 / R2;
    real_t invR = Bi->SRC * Bj->SRC * sqrt(invR2);
    dX *= invR2 * invR;
    pot += invR;
    ax += dX[0];
    ay += dX[1];
    az += dX[2];
    Bi.attr()[0] += pot;
    Bi.attr()[1] -= dX[0];
    Bi.attr()[2] -= dX[1];
    Bi.attr()[3] -= dX[2];
    if (Bi != Bj) {
      Bj.attr()[0] += invR;
      Bj.attr()[1] += dX[0];
      Bj.attr()[2] += dX[1];
      Bj.attr()[3] += dX[2];
    }
  }
}

#endif /*TAPAS_USE_VECTORMAP*/
