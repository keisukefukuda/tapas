/* vectormap_cpu.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_CPU_H_
#define TAPAS_VECTORMAP_CPU_H_

/** @file vectormap_cpu.h @brief A mock direct part working similar
    way to the GPU implementation. */

/* NOTES: This is a dummy; See "map.h", where the mappings are
   open-coded.  This class is a selector of mappings on bodies:
   vector_map1() and vector_map2().  It is not parametrized on the
   "Cell" type, because it is not available yet at the use of this
   class. */

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
};

}

#endif /*TAPAS_VECTORMAP_CPU_H_*/
