/**
 * Test for developing SPH applications.
 * (not SPH itself)
 */

#ifndef TAPAS_DEBUG
# define TAPAS_DEBUG 1 // force define TAPAS_DEBUG
#endif

#include <stdint.h>
#include <mpi.h>

#include <tapas.h>
#include <tapas/hot.h>
#include <tapas/test.h>
#include <tapas/mpi_util.h>

SETUP_TEST;

const constexpr double LatWidth = 0.1;

enum class BodyType {
  Corner,
  Surface,
  Edge,
  Other,
};

struct CellAttr {
};

template<int DIM>
struct Body {
  double pos[DIM];
  BodyType type;
};

struct BodyAttr {
  int count; // Count neighboring bodies
};

template<int DIM>
struct Params: public tapas::HOT<DIM, double, Body<DIM>, 0, BodyAttr, CellAttr>{};

template<int DIM>
using T = tapas::Tapas<Params<DIM>>;

template<int DIM>
struct Traversal {
  template<class Cell>
  inline void operator()(Cell &c1, const Cell &c2) {
    double d = std::sqrt(T<2>::Distance2(c1, c2, tapas::Shortest));
    
    if (d > LatWidth * 1.001) {
      return;
    }
    
    if (c1.IsLeaf() && c2.IsLeaf()) {
      T<DIM>::Map(*this, tapas::Product(c1.bodies(), c2.bodies()));
    } else if (!c1.IsLeaf() && !c2.IsLeaf()) {
      T<DIM>::Map(*this, tapas::Product(c1.subcells(), c2.subcells()));
    } else if (c1.IsLeaf()) {
      T<DIM>::Map(*this, tapas::Product(c1, c2.subcells()));
    } else {
      T<DIM>::Map(*this, tapas::Product(c1.subcells(), c2));
    }
  }
  
  template<class _Body, class _BodyAttr>
  inline void operator()(_Body &b1, _BodyAttr &a1,
                         const _Body &b2, const _BodyAttr &) {
    double d = std::sqrt(T<DIM>::Distance2(b1, b2));
    //std::cout << "Particle to Particle: distance = " << d << " (while LatWidth=" << LatWidth << ")" << std::endl;
    if (d < LatWidth * 1.001) {
      a1.count++;
    }
  }
};


void TestLatticeCount(int N, int ncrit) {
  // 2-dim tests
  std::vector<Body<2>> bodies;
  std::vector<BodyAttr> attrs;

  using T2 = T<2>;
  //using T3 = T<3>;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Body<2> b;
      b.pos[0] = LatWidth * i;
      b.pos[1] = LatWidth * j;

      // Corner
      if ((i == 0 || i == N-1) && (j == 0 || j == N-1)) {
        b.type = BodyType::Corner;
      } else if (i == 0 || i == N-1 || j == 0 || j == N-1) {
        b.type = BodyType::Edge;
      } else {
        b.type = BodyType::Other;
      }
          
      bodies.push_back(b);

      BodyAttr a = {0};
      attrs.push_back(a);
    }
  }

  auto *root = T2::Partition(bodies, attrs, ncrit);
  T2::Map(Traversal<2>(), tapas::Product(*root, *root));

  size_t nb = root->GetBodies().size();
  if (tapas::mpi::Size() == 1) {
    ASSERT_EQ(N*N, nb);
  }

  for (size_t i = 0; i < nb; i++) {
    auto &b = root->GetBodies()[i];
    auto &a = root->GetBodyAttrs()[i];
    switch(b.type) {
      case BodyType::Corner: ASSERT_EQ(3, a.count); break;
      case BodyType::Edge:   ASSERT_EQ(4, a.count); break;
      case BodyType::Other:  ASSERT_EQ(5, a.count); break;
      default:assert(0);
    }
  }

  delete root;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const std::vector<int> kNcrits = {4, 16, 32};

  // smaller scale tests with 1 process
  //TestLatticeCount(5, 16); => assertion fail
  TestLatticeCount(10, 16);
  
  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}

