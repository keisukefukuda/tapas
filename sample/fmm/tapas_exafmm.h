#ifndef EXAFMM_TAPAS_H_
#define EXAFMM_TAPAS_H_

#include <cstddef> // offsteOf

#include "types.h" // exafmm/include/types.h
#include "tapas/debug_util.h"
#include "tapas/util.h"

#include "tapas.h"

#ifdef FMM_MUTUAL
# define _CONST
#else
# define _CONST const
#endif

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

// Body is defined in types.h
using BodyAttr = kvec4;

#include "tapas/hot.h"

#ifdef MTHREADS
#warning "MTHREADS is defined. Do you mean \"MTHREAD\"?"
#endif

// Select threading component: serial/MassiveThreads/TBB
#if defined(MTHREAD)

#include "tapas/threading/massivethreads.h"
using FMM_Threading = tapas::threading::MassiveThreads;

#elif defined(TBB)

#include "tapas/threading/tbb.h"
using FMM_Threading = tapas::threading::IntelTBB;

#else

#include "tapas/threading/serial.h"
using FMM_Threading = tapas::threading::Serial;

#endif

struct FMM_Params : public tapas::HOT<3, real_t, Body, offsetof(Body, X), kvec4, CellAttr> {
  using Threading = FMM_Threading;
};

using TapasFMM = tapas::Tapas<FMM_Params>;

typedef TapasFMM::Region Region;

#endif // EXAFMM_TAPAS_H_
