/* vectormap_cpu.h -*- Mode: C++; Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2017 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_CPU_H_
#define TAPAS_VECTORMAP_CPU_H_

#include "vectormap_util.h"

/** @file vectormap_cpu.h @brief A mock direct part working similar
    way to the GPU implementation. */

/* NOTES: (0) This class selects an implementation of mapping on
   bodies.  It is not parametrized on the "Cell" type, because it is
   not available yet at the use of this class.  (1) It assumes some
   operations defined in the underlying datatypes.  The datatype of
   body attribute should define initializer by a constant (value=0.0)
   and assignment by "+=".  (2) It iterates over the bodies by
   "BodyIterator" directly, not via "ProductIterator".  (3) The code
   is fixed on the current definition of "AllowMutualInteraction()".
   The code should be fixed as it changes. */

#ifdef TAPAS_COMPILER_INTEL
#define TAPAS_FORCEINLINE _Pragma("forceinline")
#else
#define TAPAS_FORCEINLINE
#endif

namespace tapas {

template<int _DIM, class _FP, class _BT, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CPU {
  template <typename T>
  using um_allocator = std::allocator<T>;

  static void vectormap_setup(int /* cta */, int /* nstreams */) {}
  static void vectormap_release() {}

  static void vectormap_start() {}

  template<class Funct, class Cell, class ...Args>
  static void vectormap_finish(Funct, Cell, Args...) {}

  void Setup(int cta, int nstreams) {}
  void Start2() {}
  void Finish2() {}

  template <class Fn, class Cell, class...Args>
  void vmap1(Fn f, BodyIterator<Cell> iter, Args... args) {
    for (size_t i = 0; i < iter.size(); ++i) {
      f(iter.cell(), *iter, iter.attr(), std::forward<Args>(args)...);
      iter++;
    }
  }

  /* This was ProductMapImpl() in hot/mapper.h. */

  template <class Fn, class CELL, class...Args>
  void vmap2(Fn f, ProductIterator<BodyIterator<CELL>> prod,
             bool mutualinteraction, Args&&... args) {
    //using BodyIterator = typename CELL::BodyIterator;
    typename CELL::BodyIterator iter1 = prod.t1_;
    int beg1 = 0;
    int end1 = prod.t1_.size();
    typename CELL::BodyIterator iter2 = prod.t2_;
    int beg2 = 0;
    int end2 = prod.t2_.size();

    TAPAS_ASSERT(beg1 < end1 && beg2 < end2);

    using Body = typename CELL::Body;
    using BodyAttr = typename CELL::BodyAttr;
    bool mutual = mutualinteraction && (iter1.cell() == iter2.cell());

    CELL &c1 = iter1.cell();
    CELL &c2 = iter2.cell();

    c1.WeightLf((end1 - beg1) * (end2 - beg2));
    if (mutual) {
      c2.WeightLf((end1 - beg1) * (end2 - beg2));
    }

    auto *bodies1 = &c1.body(0);
    auto *bodies2 = &c2.body(0);
    auto *attrs1 = &c1.body_attr(0);
    auto *attrs2 = &c2.body_attr(0);

    if (mutual) {
      for (int i = beg1; i < end1; i++) {
        for (int j = beg2; j <= i; j++) {
          if (1) {
            TAPAS_FORCEINLINE
              f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
          }
        }
      }
    } else {
      for (int i = beg1; i < end1; i++) {
        for (int j = beg2; j < end2; j++) {
          TAPAS_FORCEINLINE
            f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
        }
      }
    }
  }
};

}

#endif /*TAPAS_VECTORMAP_CPU_H_*/
