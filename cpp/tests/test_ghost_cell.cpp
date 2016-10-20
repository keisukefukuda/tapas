
#include <cmath>
#include <utility>
#include <set>

#ifndef TAPAS_DEBUG
#define TAPAS_DEBUG 1 // always use TAPAS_DEBUG
#endif

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/basic_types.h>
#include <tapas/hot/ghost_cell.h>

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

template<int Dim, class FP>
auto make_cell(std::initializer_list<FP> min, std::initializer_list<FP> max, FP width) {
  using R = tapas::Region<Dim, FP>;
  using G = tapas::hot::GhostCell<R>;
  return G(R(tapas::Vec<Dim, FP>(min), tapas::Vec<Dim, FP>(max)), width);
}

void Test_Distance_Center_1d() {
  using FP = double;

  auto make_cell_1d = make_cell<1, FP>;

  {
    // Two regions are separated.
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                                  |----------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0);
    auto gc2 = make_cell_1d({ 2.0}, { 4.0}, 2.0);

    ASSERT_TRUE(Close(gc1.Distance(gc2, tapas::CenterClass()), 3.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.Distance(gc1, tapas::CenterClass()), 3.0, __FILE__, __LINE__));

    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0);
    auto gc4 = make_cell_1d({ 2.0}, { 4.0}, 1.0);
    ASSERT_TRUE(Close(gc3.Distance(gc4, tapas::CenterClass()), 2.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.Distance(gc3, tapas::CenterClass()), 2.0, __FILE__, __LINE__));
  }

  {
    // Two regions are neighboring
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                            |-------------------------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0);
    auto gc2 = make_cell_1d({ 1.0}, { 3.0}, 2.0);
    
    ASSERT_TRUE(Close(gc1.Distance(gc2, tapas::CenterClass()), 2.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.Distance(gc1, tapas::CenterClass()), 2.0, __FILE__, __LINE__));


    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0);
    auto gc4 = make_cell_1d({ 1.0}, { 3.0}, 1.0);
    ASSERT_TRUE(Close(gc3.Distance(gc4, tapas::CenterClass()), 1.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.Distance(gc3, tapas::CenterClass()), 1.0, __FILE__, __LINE__));
  }
  
  {
    // Two regions are overlapping
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                       |-------------------------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0);
    auto gc2 = make_cell_1d({ 0.0}, { 2.0}, 2.0);
    
    ASSERT_TRUE(Close(gc1.Distance(gc2, tapas::CenterClass()), 1.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.Distance(gc1, tapas::CenterClass()), 1.0, __FILE__, __LINE__));

    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0);
    auto gc4 = make_cell_1d({ 0.0}, { 2.0}, 1.0);
    ASSERT_TRUE(Close(gc3.Distance(gc4, tapas::CenterClass()), 0.0, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.Distance(gc3, tapas::CenterClass()), 0.0, __FILE__, __LINE__));
  }
}

void Test_Distance_Center_2d() {
  // See cpp/tests/geometry_test.xlsx
  
  using FP = double;
  auto make_cell_2d = make_cell<2, FP>;

  // test 1
  auto A = make_cell_2d({-5.0, 0.0}, {-1.0, 4.0}, 1.0);
  auto B = make_cell_2d({ 1.0, 0.0}, { 6.0, 5.0}, 1.0);

  ASSERT_CLOSE(A.Distance(A, tapas::CenterClass()), 0.0);
  
  ASSERT_CLOSE(A.Distance(B, tapas::CenterClass()), 3.0);
  ASSERT_CLOSE(B.Distance(A, tapas::CenterClass()), 3.0);

  // test 2
  auto C = make_cell_2d({ 1.0, -2.0}, { 5.0, 2.0}, 1.0);
  ASSERT_CLOSE(A.Distance(C, tapas::CenterClass()), 3.0);
  ASSERT_CLOSE(C.Distance(A, tapas::CenterClass()), 3.0);

  // test 3
  auto D = make_cell_2d({ 1.0, -4.0}, { 5.0, 0.0}, 1.0);
  double ans3 = 3.16227766;
  ASSERT_CLOSE(A.Distance(D, tapas::CenterClass()), ans3);
  ASSERT_CLOSE(D.Distance(A, tapas::CenterClass()), ans3);

  // test 4
  auto E = make_cell_2d({ 1.0, -5.0}, { 5.0,-1.0}, 1.0);
  double ans4 = 3.605551275;
  ASSERT_CLOSE(A.Distance(E, tapas::CenterClass()), ans4);
  ASSERT_CLOSE(E.Distance(A, tapas::CenterClass()), ans4);
  
  // test 5
  auto F = make_cell_2d({ 1.0, -2.0}, { 3.0, 0.0}, 2.0);
  double ans5 = 3.807886553;
  ASSERT_CLOSE(A.Distance(F, tapas::CenterClass()), ans5);
  ASSERT_CLOSE(F.Distance(A, tapas::CenterClass()), ans5);
  
  // test 6
  auto G = make_cell_2d({ 1.0, -1.0}, { 3.0, 1.0}, 2.0);
  double ans6 = 3.535533906;
  ASSERT_CLOSE(A.Distance(G, tapas::CenterClass()), ans6);
  ASSERT_CLOSE(G.Distance(A, tapas::CenterClass()), ans6);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Distance_Center_1d();
  Test_Distance_Center_2d();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
