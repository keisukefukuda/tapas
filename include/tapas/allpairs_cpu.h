/* allpairs_cpu.h -*- Mode: C++; Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2017 RIKEN AICS */

/* Tapas All-pairs */

#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

#ifdef TAPAS_COMPILER_INTEL
#define TAPAS_FORCEINLINE _Pragma("forceinline")
#else
#define TAPAS_FORCEINLINE
#endif

#define BR0_ {
#define BR1_ }

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

namespace tapas BR0_

/*
 | fun allpairs H G Z X Y W =
 |    map (fn (xi, wi) =>
 |            H (foldl (fn (yi, a) => (G xi yi a)) Z Y) wi)
 |        (zip X W)
*/

struct TESLA_cpu {};

template <class FnH, class FnG, class Z,
          class VecR, class VecX, class VecY, class VecW>
__forceinline__
static void
allpairs(TESLA_cpu& tesla,
         int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
         FnH h, FnG g, Z zero, VecR r, VecX x, VecY y, VecW w) {
    using X = typename VecX::value_type;
    using Y = typename VecY::value_type;
    using W = typename VecW::value_type;

    int xsz = (int)x.size();
    int ysz = (int)y.size();
    assert(xsz != 0 && ysz != 0);

    for (int i = 0; i < xsz; i++) {
        X& xi = x[i];
        W& wi = w[i];
        Z acc;
        acc = zero;
        for (int j = 0; j < ysz; j++) {
            Y& yj = y[j];
            TAPAS_FORCEINLINE
                acc = g(xi, yj, wi, acc);
        }
        if (r != 0) {
            r[i] = h(wi, acc);
        }
    }
}

BR1_

//#undef TAPAS_CEILING
//#undef TAPAS_FLOOR

#undef BR0_
#undef BR1_

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
