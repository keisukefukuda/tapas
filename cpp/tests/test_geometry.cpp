
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

bool Close(double a, double b, const char *, int) {
  return fabs(a - b) < 1e-6;
}

bool Close(double a, V1 b, const char *, int) {
  return fabs(a - b[0]) < 1e-6;
}

bool Close(V1 a, double b, const char *, int) {
  return fabs(a[0] - b) < 1e-6;
}

bool Close(V1 a, V1 b, const char *file, int line) {
  bool ret = fabs(a[0] - b[0]) < 1e-10;
  if (!ret) {
    std::cerr << file << ":" << line << " Close(): Not close: a = " << a << ", b = " << b << std::endl;
  }
  return ret;
}
void Test_Separated() {
  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(!tapas::Separated(x, x));
    ASSERT_TRUE(!tapas::Separated(y, y));
  }

  {
    V2 xmax = { 1, 1};
    V2 xmin = { 0, 0};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V2 xmax = { 2, 0};
    V2 xmin = { 1,-1};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V2 xmax = { 1, 1};
    V2 xmin = { -0.1, -0.1};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(!tapas::Separated(x, y));
    ASSERT_TRUE(!tapas::Separated(y, x));
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Separated();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
