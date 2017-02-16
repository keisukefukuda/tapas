
#include <cmath>
#include <utility>
#include <set>
#include <type_traits>


#ifndef TAPAS_DEBUG
#define TAPAS_DEBUG 1 // always use TAPAS_DEBUG
#endif

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/basic_types.h>

#include <tapas/hot.h>
#include <tapas/hot/proxy/proxy_cell.h>
#include <tapas/hot/proxy/oneside_traverse_policy.h>

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

struct Cell { using KeyType = unsigned long long; };
struct Body {};
struct BodyAttr {};

struct Data {
  using KeyType = uint64_t;
  using CellType = Cell;
  using BodyType = Body;
  using BodyAttrType = BodyAttr;
  int max_depth_;
};

using FP = double;
using Region1 = tapas::Region<1, FP>;
using Region2 = tapas::Region<2, FP>;

using Vec1 = tapas::Vec<1, FP>;
using Vec2 = tapas::Vec<2, FP>;

template<int Dim, class FP>
auto make_cell(std::initializer_list<FP> min_,
               std::initializer_list<FP> max_,
               const tapas::Vec<Dim, FP>& width, int depth) {
  using Policy = tapas::hot::proxy::OnesideTraversePolicy<Dim, FP, Data>;
  //static_assert(std::is_same<typename Policy::Reg, Region1>::value, "");

  static Data data = {5};
  using V = tapas::Vec<Dim, FP>;
  using R = tapas::Region<Dim, FP>;
  
  return Policy(data, R(V(min_), V(max_)), width, depth);
}

void Test_Distance_Center_1d() {
  auto make_cell_1d = make_cell<1, double>;
  double ans;

  {
    // Two regions are separated.
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                                  |----------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0, 1);
    auto gc2 = make_cell_1d({ 2.0}, { 4.0}, 2.0, 1);
    ans = 3.0;
    ASSERT_TRUE(Close(gc1.dX(gc2, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.dX(gc1, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));

    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0, 1);
    auto gc4 = make_cell_1d({ 2.0}, { 4.0}, 1.0, 1);
    ans = 2.0;
    ASSERT_TRUE(Close(gc3.dX(gc4, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.dX(gc3, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
  }

  {
    // Two regions are neighboring
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                            |-------------------------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0, 1);
    auto gc2 = make_cell_1d({ 1.0}, { 3.0}, 2.0, 1);
    ans=2.0;
    ASSERT_TRUE(Close(gc1.dX(gc2, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.dX(gc1, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));


    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0, 1);
    auto gc4 = make_cell_1d({ 1.0}, { 3.0}, 1.0, 1);
    ans=1.0;
    ASSERT_TRUE(Close(gc3.dX(gc4, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.dX(gc3, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
  }
  
  {
    // Two regions are overlapping
    //
    // - -------------------0-----1-----2----------------------------> +X
    //
    //                 |----------| Region 1
    //                       |-------------------------| Region 2
    auto gc1 = make_cell_1d({-1.0}, { 1.0}, 2.0, 1);
    auto gc2 = make_cell_1d({ 0.0}, { 2.0}, 2.0, 1);
    ans=1.0;
    ASSERT_TRUE(Close(gc1.dX(gc2, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc2.dX(gc1, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));

    // if ghost cell width is 1.0 each.
    auto gc3 = make_cell_1d({-1.0}, { 1.0}, 1.0, 1);
    auto gc4 = make_cell_1d({ 0.0}, { 2.0}, 1.0, 1);
    ans=0.0;
    ASSERT_TRUE(Close(gc3.dX(gc4, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
    ASSERT_TRUE(Close(gc4.dX(gc3, tapas::Center).norm(), ans*ans, __FILE__, __LINE__));
  }
}

void Test_Distance_Center_2d() {
  // See cpp/tests/geometry_test.xlsx
  auto make_cell_2d = make_cell<2, FP>;

  // test 1
  auto A = make_cell_2d({-5.0, 0.0}, {-1.0, 4.0}, 1.0, 1);
  auto B = make_cell_2d({ 1.0, 0.0}, { 6.0, 5.0}, 1.0, 1);

  ASSERT_CLOSE(A.dX(A, tapas::Center).norm(), 0.0*0.0);
  
  ASSERT_CLOSE(A.dX(B, tapas::Center).norm(), 3.0*3.0);
  ASSERT_CLOSE(B.dX(A, tapas::Center).norm(), 3.0*3.0);

  // test 2
  auto C = make_cell_2d({ 1.0, -2.0}, { 5.0, 2.0}, 1.0, 1);
  ASSERT_CLOSE(A.dX(C, tapas::Center).norm(), 3.0*3.0);
  ASSERT_CLOSE(C.dX(A, tapas::Center).norm(), 3.0*3.0);

  // test 3
  auto D = make_cell_2d({ 1.0, -4.0}, { 5.0, 0.0}, 1.0, 1);
  double ans3 = 3.16227766;
  ASSERT_CLOSE(A.dX(D, tapas::Center).norm(), ans3*ans3);
  ASSERT_CLOSE(D.dX(A, tapas::Center).norm(), ans3*ans3);

  // test 4
  auto E = make_cell_2d({ 1.0, -5.0}, { 5.0,-1.0}, 1.0, 1);
  double ans4 = 3.605551275;
  ASSERT_CLOSE(A.dX(E, tapas::Center).norm(), ans4*ans4);
  ASSERT_CLOSE(E.dX(A, tapas::Center).norm(), ans4*ans4);
  
  // test 5
  auto F = make_cell_2d({ 1.0, -2.0}, { 3.0, 0.0}, 2.0, 1);
  double ans5 = 3.807886553;
  ASSERT_CLOSE(A.dX(F, tapas::Center).norm(), ans5*ans5);
  ASSERT_CLOSE(F.dX(A, tapas::Center).norm(), ans5*ans5);
  
  // test 6
  auto G = make_cell_2d({ 1.0, -1.0}, { 3.0, 1.0}, 2.0, 1);
  double ans6 = 3.535533906;
  ASSERT_CLOSE(A.dX(G, tapas::Center).norm(), ans6*ans6);
  ASSERT_CLOSE(G.dX(A, tapas::Center).norm(), ans6*ans6);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Distance_Center_1d();
  Test_Distance_Center_2d();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
