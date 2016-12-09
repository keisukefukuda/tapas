/**
 * Test of:
 * (1) When Bodies and Body Attributes are given as the initial values, check a pair of body and its attributes always match.
 */

#include <cstdlib>

#ifndef TAPAS_DEBUG
# define TAPAS_DEBUG 1// force define TAPAS_DEBUG
#endif

#include <mpi.h>

#include <tapas.h>
#include <tapas/hot.h>
#include <tapas/test.h>
#include <tapas/mpi_util.h>

SETUP_TEST;

const constexpr int Dim = 3;

struct CellAttr1 { };

struct BodyAttr1 {
  int val;
};

struct Body1 {
  double pos[Dim];
  int id; // should be the same id to the correspoinding body object.
  int val;
};

struct TestParams1 : public tapas::HOT<Dim, double, Body1, 0, BodyAttr1, CellAttr1> { };

using TapasTest1 = tapas::Tapas<TestParams1>;

struct TestTraverse1 {
  template<class Cell>
  inline void operator()(Cell &, Cell &C, std::vector<int> &check) {
    if (C.IsLeaf()) {
      TapasTest1::Map(*this, C.bodies(), check);
    } else {
      TapasTest1::Map(*this, C.subcells(), check);
    }
  }

  template<class Cell>
  inline void operator()(Cell &, Body1 &b, BodyAttr1 &a, std::vector<int> &check) {
    if (!Cell::Inspector) {
      check[b.id] = (b.val == a.val);
      //ASSERT_EQ(true, b.val == a.val);
    }
    if (b.val != a.val) {
      std::cerr << "Error: b.val != a.val: (b.val = " << b.val << ", a.val = " << a.val << ")" << std::endl;
    }
  }
};

void Test_1_Init() {
  // Test if Body and BodyAttr data is initialized correctly.

  // TODO: If we ncrit 1 or 2, the test program crashes. need to debug.
  const std::vector<int> kNcrits = {4, 16, 32};
  for (int ncrit : kNcrits) {
    const int nb = 1000;

    std::vector<Body1> bodies(nb);
    std::vector<BodyAttr1> attrs(bodies.size());
    std::vector<int> check(nb, 0);

    srand(0);

    for (int i = 0; i < bodies.size(); i++) {
      bodies[i].pos[0] = drand48() - 0.5;
      bodies[i].pos[1] = drand48() - 0.5;
      bodies[i].pos[2] = drand48() - 0.5;
      bodies[i].id = i;
      bodies[i].val = attrs[i].val = i * i;
    }

    assert(bodies.size() == attrs.size());

    TapasTest1::Cell *root = TapasTest1::Partition(bodies.data(), attrs.data(), bodies.size(), ncrit, MPI_COMM_WORLD);
    assert(root);

    auto func = TestTraverse1();

    if (root->IsLeaf()) {
      TapasTest1::Map(func, root->bodies(), check);
    } else {
      TapasTest1::Map(func, root->subcells(), check);
    }
    TapasTest1::Destroy(root);

    std::vector<int> reduced;
    tapas::mpi::Reduce(check, reduced, MPI_LOR, 0, MPI_COMM_WORLD);
    
    // check the value
    if (tapas::mpi::Rank() == 0) {
      for (size_t i = 0; i < reduced.size(); i++) {
        ASSERT_EQ(true, reduced[i]);
      }
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_1_Init();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
