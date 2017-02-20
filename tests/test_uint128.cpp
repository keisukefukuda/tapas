/**
 * Test of:
 * Sum up values of all particles
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
#include <tapas/uint128.h>

SETUP_TEST;

void TestUint128() {
  uint128_t one = 1;
  uint128_t two = 2;
  uint128_t three = 3;

  ASSERT_EQ(one + two, three);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  TestUint128();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}

  
