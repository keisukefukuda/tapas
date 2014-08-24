#ifndef TACO_H_
#define TACO_H_

#if !defined(PARTICLE_TYPE)
#error PARTICLE_TYPE not defined
#endif
#if !defined(CELL_TYPE)
#error CELL_TYPE not defined
#endif

PARTICLE_TYPE;
CELL_TYPE;

#include "primitive_types.h"
#include "vec.h"

#define DeclareRegionType(dim, key) \
  typedef struct region##dim##key { \
    vec##dim##key begin;            \
    vec##dim##key end;              \
  } region##dim##key;
DeclareRegionType(1, f)
DeclareRegionType(2, f)
DeclareRegionType(3, f)
DeclareRegionType(1, d)
DeclareRegionType(2, d)
DeclareRegionType(3, d)
#if defined(DEFAULT_FP_TYPE_FLOAT)
#define region1r region1f
#define region2r region2f
#define region3r region3f
#elif defined(DEFAULT_FP_TYPE_DOUBLE)
#define region1r region1d
#define region2r region2d
#define region3r region3d
#endif

extern int is_leaf(const CELL_TYPE * const); 
extern index_t cell_np(const CELL_TYPE * const);
extern real_t cell_distance(const CELL_TYPE * const c1, const CELL_TYPE * const c2);
extern void map(void *f, void *c, ...);
extern void *product(const void *cl1,  const void *cl2);
extern void *get_force(const void *c);
extern PARTICLE_TYPE get_particle(const CELL_TYPE * const c, index_t idx);
extern void *get_subcells(const void *c);
extern CELL_TYPE *get_subcell(const void *c, int idx);
extern void accumulate_force1f(const void *c, size_t idx, vec1f f);
extern void accumulate_force2f(const void *c, size_t idx, vec2f f);
extern void accumulate_force3f(const void *c, size_t idx, vec3f f);
extern void accumulate_force1d(const void *c, size_t idx, vec1d f);
extern void accumulate_force2d(const void *c, size_t idx, vec2d f);
extern void accumulate_force3d(const void *c, size_t idx, vec3d f);

#if defined(DEFAULT_FP_TYPE_FLOAT)
#define accumulate_force1r accumulate_force1f
#define accumulate_force2r accumulate_force2f
#define accumulate_force3r accumulate_force3f
#elif defined(DEFAULT_FP_TYPE_DOUBLE)
#define accumulate_force1r accumulate_force1d
#define accumulate_force2r accumulate_force2d
#define accumulate_force3r accumulate_force3d
#endif

extern CELL_TYPE *partition_bsp1f(void *p, size_t np, region1f r, int s);
extern CELL_TYPE *partition_bsp2f(void *p, size_t np, region2f r, int s);
extern CELL_TYPE *partition_bsp3f(void *p, size_t np, region3f r, int s);
extern CELL_TYPE *partition_bsp1d(void *p, size_t np, region1d r, int s);
extern CELL_TYPE *partition_bsp2d(void *p, size_t np, region2d r, int s);
extern CELL_TYPE *partition_bsp3d(void *p, size_t np, region3d r, int s);
#if defined(DEFAULT_FP_TYPE_FLOAT)
#define partition_bsp1r partition_bsp1f
#define partition_bsp2r partition_bsp2f
#define partition_bsp3r partition_bsp3f
#elif defined(DEFAULT_FP_TYPE_DOUBLE)
#define partition_bsp1r partition_bsp1d
#define partition_bsp2r partition_bsp2d
#define partition_bsp3r partition_bsp3d
#endif

#if DEFAULT_NUM_DIM == 1
#define accumulate_force accumulate_force1r
#define partition_bsp partition_bsp1r
#elif DEFAULT_NUM_DIM == 2
#define accumulate_force accumulate_force2r
#define partition_bsp partition_bsp2r
#elif DEFAULT_NUM_DIM == 3
#define accumulate_force accumulate_force3r
#define partition_bsp partition_bsp3r
#endif

#endif /* TACO_H_ */
