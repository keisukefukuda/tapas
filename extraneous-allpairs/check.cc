/* check.cc (2015-10-28) -*-Coding: us-ascii-unix;-*- */

#include <vector>
#include <stdio.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__
#endif

#include "tapar-tuple.h"
#include <vector>

int
main(int argc, char **argv) {
    int a0 = 10;
    double a1 = 20.0;
    tapar::tuple<int, double> a(a0, a1);
    int b0 = 30;
    double b1 = 40.0;
    tapar::tuple<int, double> b(b0, b1);
    a = b;
    printf("a.0=%d a.1=%f\n", tapar::get<0>(a), tapar::get<1>(a));
    printf("a0=%d a1=%f\n", a0, a1);

    std::vector<int> v0(10);
    std::vector<double> v1(10);
    tapar::zipped<int, double> z (v0.data(), v1.data());

    tapar::tuple<int, double> tx = z[4];
    tx = b;
    printf("z[4].0=%d z[4].1=%f\n", tapar::get<0>(z[4]), tapar::get<1>(z[4]));

    tapar::vector<int> x (v0.data(), v0.size());
    printf("x[4]=%d\n", x[4]);
    x[4] = 50;
    printf("x[4]=%d\n", x[4]);

    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: t
// End:
