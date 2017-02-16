
#include <cmath>
#include <utility>
#include <set>

#ifndef TAPAS_DEBUG
#define TAPAS_DEBUG 1 // always use TAPAS_DEBUG
#endif

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/geometry.h>

SETUP_TEST;

using V1 = tapas::Vec<1, double>;
using V2 = tapas::Vec<2, double>;
using Reg1 = tapas::Region<1, double>;
using Reg2 = tapas::Region<2, double>;

void Test_Join() {
  {
    const int Dim = 1;
    using FP = double;
    using Region = tapas::Region<Dim, FP>;

    auto A = Region::BB(Region({1}, {2}), Region({-1}, {3}));
    ASSERT_EQ(A.min(0), -1);
    ASSERT_EQ(A.max(0), 3);

    auto B = Region::BB(Region({1}, {2}), Region({3}, {4}));
    ASSERT_EQ(B.min(0), 1);
    ASSERT_EQ(B.max(0), 4);
  }
  
  {
    const int Dim = 2;
    using FP = double;
    using Region = tapas::Region<Dim, FP>;

    auto C = Region::BB(Region({1,1}, {2,2}), Region({-1,-1}, {3,3}));
    ASSERT_EQ(C.max(0), 3);
    ASSERT_EQ(C.max(1), 3);
    ASSERT_EQ(C.min(0), -1);
    ASSERT_EQ(C.min(1), -1);
  }
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Join();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
