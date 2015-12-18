#ifndef EXAFMM_TAPAS_H_
#define EXAFMM_TAPAS_H_

#include "types.h" // exafmm/include/types.h
#include "tapas/debug_util.h"

#include "tapas.h"

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

typedef tapas::BodyInfo<Body, 0> BodyInfo;

// Select MPI or single-process
#ifdef USE_MPI

// Use multi-process version of hashed octree
#include "tapas/hot.h"
using HOT = tapas::HOT<3, tapas::sfc::Morton>;

#else /* USE_MPI */

// Use single-process version of hashed octree
#include "tapas/single_node_hot.h"
using HOT = tapas::SingleNodeHOT<3, tapas::sfc::Morton>;

#endif /* USE_MPI */


// Select threading component: serial or MassiveThreads
#if defined(MTHREAD) || defined(MTHREADS)
#include "tapas/threading/massivethreads.h"
using Threading = tapas::threading::MassiveThreads;
#else /* no MTHREADS */
using Threading = tapas::threading::Default;
#endif /* MTHREADS */

#if defined(__CUDACC__) && defined(TAPAS_USE_VECTORMAP)
using VectormapImpl = tapas::Vectormap_CUDA_Packed<3, real_t, BodyInfo, kvec4>;
#else
using VectormapImpl = tapas::Vectormap_CPU<3, real_t, BodyInfo, kvec4>;
#endif /*defined(__CUDACC__) && defined(TAPAS_USE_VECTORMAP)*/

typedef tapas::Tapas<3, real_t, BodyInfo, kvec4, CellAttr,
                     HOT, Threading, VectormapImpl> Tapas;

typedef Tapas::Region Region;

#if 0 // to be deleted 
namespace tapas_kernel {

void P2M(Tapas::Cell &C);
void M2M(Tapas::Cell &C);
void M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, vec3 Xperiodic, bool mutual);
void L2L(Tapas::Cell &C);
void L2P(Tapas::BodyIterator &B);
void P2P(Tapas::BodyIterator &Ci, Tapas::BodyIterator &Cj, vec3 Xperiodic);

} // tapas_kernel
#endif /* if 0 */

#endif // EXAFMM_TAPAS_H_
