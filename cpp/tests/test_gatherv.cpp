#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>

#include <mpi.h>

#include <tapas/mpi_util.h>
#include <tapas/test.h>

SETUP_TEST;

// test 1 : Send custom struct.
struct T {
  int rank;
  int rank2;
  bool operator==(const T& rhs) const {
    return rank == rhs.rank && rank2 == rhs.rank2;
  }
};

std::ostream& operator<<(std::ostream &os, const T& t) {
  os << "{" << t.rank << "," << t.rank2 << "}";
  return os;
}

void Test_Gatherv() {
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Test MPI_Gatherv.
  {
    for (int root = 0; root < size; root++) {
      // test 1 : rank i sends [i,i,..,i] (length = i)
      std::vector<int> sendbuf(rank);
      std::vector<int> recvbuf;
    
      for (size_t i = 0; i < sendbuf.size(); i++) { sendbuf[i] = rank; }
      
      tapas::mpi::Gatherv(sendbuf, recvbuf, root, MPI_COMM_WORLD);
    
      if (rank == root) {
        std::vector<int> should_be;
        for (int r = 0; r < size; r++) {
          for (int i = 0; i < r; i++) {
            should_be.push_back(r);
          }
        }
    
        int total = (size-1) * size / 2;
        ASSERT_EQ(total, recvbuf.size());
        ASSERT_EQ(should_be, recvbuf);
      }
    }
  }

  {
    for (int root = 0; root < size; root++) {
      // test 2 : rank i sends an array of the custom type T, of which length is i.
      std::vector<T> sendbuf(rank);
      std::vector<T> recvbuf;
      for (size_t i = 0; i < sendbuf.size(); i++) {
        T t = {rank, rank*2};
        sendbuf[i] = t;
      }

      tapas::mpi::Gatherv(sendbuf, recvbuf, root, MPI_COMM_WORLD);
    
      if (rank == root) {
        std::vector<T> should_be;
        for (int r = 0; r < size; r++) {
          for (int i = 0; i < r; i++) {
            T t = {r, r * 2};
            should_be.push_back(t);
          }
        }
    
        int total = (size-1) * size / 2;
        ASSERT_EQ(total, recvbuf.size());
        ASSERT_EQ(should_be, recvbuf);
      }
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  
  Test_Gatherv();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
