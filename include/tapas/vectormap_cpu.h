/* vectormap_cpu.h -*- Mode: C++; Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2017 RIKEN AICS */

#pragma once

#include "allpairs_cpu.h"
#include "vectormap_util.h"

/** @file vectormap_cpu.h @brief Direct part working similar way to
    the GPU implementation. */

/* NOTES: (0) This class selects an implementation of mapping on
   bodies.  It is not parameterized on the "Cell" type, because it is
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

#define BR0_ {
#define BR1_ }

namespace tapas BR0_

#define ALLPAIRS 1

/* Vector interface (minimal). */

template <class T>
struct Vector_Raw_cpu {
    using value_type = T;

    T* v;
    size_t bodies;

    __forceinline__
    Vector_Raw_cpu(T* v, size_t size) {
        this->v = v;
        this->bodies = size;
    }

    __forceinline__
    size_t size() {
        return bodies;
    }

    __forceinline__
    T& operator[] (size_t index) {
        return v[index];
    }
};

/* Arguments permutation for the allpairs interface. */

template <class Fn>
struct P2PFnF_cpu {
    Fn f_;

    P2PFnF_cpu(Fn f) : f_ (f) {}

    template<typename Body, typename Attr>
    __forceinline__
    Attr operator()(Body& xi, Body& yj, Attr& wi, Attr zero) {
        vec3 Xperiodic = 0;
        f_(xi, wi, yj, zero, Xperiodic);
        return zero;
    }
};

/* Dummy function for h(wi, acc). */

struct P2PFnG_cpu {
    template<typename Attr>
    __forceinline__
    Attr operator()(Attr wi, Attr acc) {
        return acc;
    }
};

template<int _DIM, class _FP, class _BODY, class _ATTR, class _CELL_ATTR>
struct Vectormap_CPU {
    template <typename T>
    using um_allocator = std::allocator<T>;

    static void vectormap_setup(int /* cta */, int /* nstreams */) {}
    static void vectormap_release() {}

    static void vectormap_start() {}

    template<class Fn, class Cell, class ...Args>
    static void vectormap_finish(Fn, Cell, Args...) {}

    void Setup(int cta, int nstreams) {}
    void Start2() {}
    void Finish2() {}

    template <class Fn, class Cell, class...Args>
    __forceinline__
    void vmap1(Fn f, BodyIterator<Cell> iter, Args... args) {
        for (size_t i = 0; i < iter.size(); ++i) {
            f(iter.cell(), *iter, iter.attr(), std::forward<Args>(args)...);
            iter++;
        }
    }

    /* This was ProductMapImpl() in hot/mapper.h. */

    template <class Fn, class CELL, class...Args>
    __forceinline__
    void vmap2(Fn f, ProductIterator<BodyIterator<CELL>> prod,
               bool mutualinteraction, Args&&... args) {
        using Body = typename CELL::Body;
        using Attr = typename CELL::BodyAttr;
        //using BodyIterator = typename CELL::BodyIterator;
        typename CELL::BodyIterator iter1 = prod.t1_;
        int beg1 = 0;
        int end1 = prod.t1_.size();
        typename CELL::BodyIterator iter2 = prod.t2_;
        int beg2 = 0;
        int end2 = prod.t2_.size();

        TAPAS_ASSERT(beg1 < end1 && beg2 < end2);

        bool mutual = mutualinteraction && (iter1.cell() == iter2.cell());

        CELL &c1 = iter1.cell();
        CELL &c2 = iter2.cell();

        c1.WeightLf((end1 - beg1) * (end2 - beg2));
        if (mutual) {
            c2.WeightLf((end1 - beg1) * (end2 - beg2));
        }

        auto* bodies1 = &c1.body(0);
        auto* bodies2 = &c2.body(0);
        auto* attrs1 = &c1.body_attr(0);
        auto* attrs2 = &c2.body_attr(0);

        if (mutual) {
            for (int i = beg1; i < end1; i++) {
                for (int j = beg2; j <= i; j++) {
                    TAPAS_FORCEINLINE
                        f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
                }
            }
        } else {

#if (ALLPAIRS == 0)

            for (int i = beg1; i < end1; i++) {
                for (int j = beg2; j < end2; j++) {
                    TAPAS_FORCEINLINE
                        f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], std::forward<Args>(args)...);
                }
            }

#else /*ALLPAIRS*/

            Vector_Raw_cpu<Body> lbody (&bodies1[beg1], (size_t)(end1 - beg1));
            Vector_Raw_cpu<Attr> lattr (&attrs1[beg1], (size_t)(end1 - beg1));
            Vector_Raw_cpu<Body> rdata (&bodies2[beg2], (size_t)(end2 - beg2));
            Attr zero = {0.0f, 0.0f, 0.0f, 0.0f};
            P2PFnF_cpu<Fn> fx (f);

            TESLA_cpu tesla;
            int tilesize = 0;
            size_t nblocks = 0;
            int ctasize = 0;
            int scratchpadsize = 0;

            allpairs(tesla,
                     tilesize, nblocks, ctasize, scratchpadsize,
                     /*h*/ P2PFnG_cpu(), /*g*/ fx, /*z*/ zero,
                     /*r*/ (Attr*)0, /*x*/ lbody, /*y*/ rdata, /*w*/ lattr);

#endif /*ALLPAIRS*/
        }
    }
};

BR1_
#undef BR0_
#undef BR1_

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
