/**
 * Test of:
 * Sum up values of all particles
 */

#ifndef TAPAS_DEBUG
# define TAPAS_DEBUG 1 // force define TAPAS_DEBUG
#endif

#include <mpi.h>

#include <tapas.h>
#include <tapas/hot.h>
#include <tapas/test.h>
#include <tapas/mpi_util.h>

SETUP_TEST;

struct CellAttr {
  double sum;
};

struct BodyAttr {
  double val;
};

template<int Dim>
struct Body {
  double pos[Dim];
};

template<int Dim>
struct Params : public tapas::HOT<Dim, double,
                                  Body<Dim>, 0,
                                  BodyAttr, CellAttr>{};

template<int Dim>
using TestTreeSum = tapas::Tapas<Params<Dim>>;

// Todo: enable using ordinary binary operator
template<class T>
struct Sum {
  void operator()(T &a, const T &b) { a += b; }
};

template<int Dim>
struct SumTraversal {
  template<class Cell>
  inline void operator()(Cell &leaf, Body<Dim> &, BodyAttr &a) {
    TestTreeSum<Dim>::Reduce(leaf, leaf.attr().sum, a.val, Sum<double>());
  }

  template<class Cell>
  inline void operator()(Cell &P, Cell &C) {
    if (C.IsLeaf()) {
      TestTreeSum<Dim>::Map(*this, C.bodies());
    } else {
      TestTreeSum<Dim>::Map(*this, C.subcells());
    }

    TestTreeSum<Dim>::Reduce(P,
                             P.attr().sum,
                             C.attr().sum,
                             Sum<double>());
  }
};

template<int Dim>
void RunTest(int num, int ncrit) {
  std::vector<Body<Dim>> bodies(num);
  std::vector<BodyAttr> attrs(num);
  double ans_sum = 0;

  for (int i = 0; i < num; i++) {
    for (int d = 0; d < Dim; d++) {
      bodies[i].pos[d] = drand48() - 0.5;
    }
    attrs[i].val = drand48();
    ans_sum += attrs[i].val;
  }
  
  auto *root = TestTreeSum<Dim>::Partition(bodies.data(), attrs.data(),
                                           bodies.size(), ncrit, MPI_COMM_WORLD);

  auto func = SumTraversal<Dim>();

  if (root->IsLeaf()) {
    TestTreeSum<Dim>::Map(func, root->bodies());
  } else {
    TestTreeSum<Dim>::Map(func, root->subcells());
  }

  double result = root->attr().sum;

  delete root;

  double diff = abs(ans_sum - result);

  if (diff / ans_sum > 1e-12) {
    std::cerr << "Error: ans_sum = " << ans_sum << ", result = " << result << " "
              << "(num=" << num << ", ncrit=" << ncrit << ", dim=" << Dim << ")"
              << std::endl;
    fprintf(stderr, "\tans_sum = %.12f\n", ans_sum);
    fprintf(stderr, "\tresult  = %.12f\n", result);
    ASSERT_TRUE(0);
  }
  ASSERT_TRUE(ans_sum - result < 1e-10);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const std::vector<int> kNcrits = {1, 16, 32};

  // smaller scale tests with 1 process
  if (tapas::mpi::Size() == 1) {
    for (int nc : kNcrits) {
      for (int num = 1; num <= 1024; num *= 2) {
        RunTest<3>(num, nc);
      }
    }
  } else {
#if 0
    for (int nc : kNcrits) {
      for (int num = 1000; num < 10000; num += 1000) {
        //RunTest<3>(num, nc);
      }
    }
#endif
  }
  
  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}

  
