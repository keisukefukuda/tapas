
#include <mutex>
#include <thread>

#include <sys/types.h>
#include <unistd.h> // usleep

#ifdef USE_MPI
# include <mpi.h>
#endif

// NOTE:
// This macro (TO_MTHREAD_NATIVE or TO_SERIAL) is needed by tpswitch.h, which included in the original ExaFMM.
// Although this is not necessary in Tapas, ExaFMM's logger class is still using it.
#if MTHREAD
# define TO_MTHREAD_NATIVE 1
# define TO_SERIAL 0
#else /* MTHREAD */
# define TO_SERIAL 1
#endif

#include "args.h"
#include "dataset.h"
#include "logger.h"
//#include "kernel.h"
//#include "up_down_pass.h"
#include "verify.h"

#include "tapas_exafmm.h"
#include "LaplaceSphericalCPU_tapas.h"
#include "LaplaceP2PCPU_tapas.h"
#include "tapasfmm_debug.h"

#ifdef TBB
# include <tbb/task_scheduler_init.h>
#endif

#ifdef COUNT /* Count kernel invocations */

# warning "COUNT is defined. This may significantly slows down execution"
uint64_t numM2L = 0;
uint64_t numP2P = 0;
inline void ResetCount() { numP2P = 0; numM2L = 0; }

#else

inline void ResetCount() { }

#endif /* ifdef COUNT */

#ifdef USE_RDTSC
# ifdef TAPAS_COMPILER_INTEL
#  define RDTSC() __rdtsc()
# endif
#else
# define RDTSC() 0
#endif

double GetTime() {
#ifdef USE_MPI
  return MPI_Wtime();
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

template <int DIM, class FP> inline
tapas::Vec<DIM, FP> &asn(tapas::Vec<DIM, FP> &dst, const vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

template <int DIM, class FP> inline
vec<DIM, FP> &asn(vec<DIM, FP> &dst, const tapas::Vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

// UpDownPass::upwardPass
struct FMM_Upward {
  template<class Cell>
  inline void operator()(Cell &parent, Cell &child, real_t theta) {
    // theta is not used now; to be deleted

#ifdef TAPAS_DEBUG_DUMP
    {
      tapas::debug::DebugStream e("FMM_Upward");
      e.out() << TapasFMM::TSP::SFC::Simplify(child.key()) << " (1) " << child.IsLeaf() << " ";
      e.out() << "child.attr().R = " << std::fixed << std::setprecision(6) << child.attr().R << " ";
      e.out() << std::endl;
    }
#endif

    CellAttr attr = child.attr();
    attr.R = 0;
    attr.M = 0;
    attr.L = 0;
    child.attr() = attr;

    // Compute the child cell recursively
    if (child.IsLeaf()) {
      TapasFMM::Map(P2M(), child.bodies()); // P2M
    } else {
      TapasFMM::Map(*this, child.subcells(), theta); // recursive
    }

    // M2M
    M2M(parent, child);
  }
  std::string label() const { return "FMM-upward"; }
};

struct FMM_Downward {
  template<class Cell>
  inline void operator()(Cell &parent, Cell &child) {
    //if (c.nb() == 0) return;

    L2L(parent, child);

    if (child.IsLeaf()) {
      if (child.nb() > 0) {
        TapasFMM::Map(L2P, child.bodies());
      }
    } else {
      TapasFMM::Map(*this, child.subcells());
    }
  }

  std::string label() const { return "FMM-downward"; }
};

// Perform ExaFMM's Dual Tree Traversal (M2L & P2P)
struct FMM_DTT {
#ifdef FMM_MUTUAL
  using P2P_Kernel = P2P_mutual;
#else
  using P2P_Kernel = P2P;
#endif
  
  std::string label() const { return "FMM-DTT"; }

  template<class Cell>
  inline void operator()(Cell &Ci, _CONST Cell &Cj, real_t theta) {
    //real_t R2 = (Ci.center() - Cj.center()).norm();
    real_t R2 = TapasFMM::Distance2(Ci, Cj, tapas::Center);
    vec3 Xperiodic = 0; // dummy; periodic is not ported

    real_t Ri = 0;
    real_t Rj = 0;

    for (int d = 0; d < 3; d++) {
      Ri = std::max(Ri, Ci.width(d));
      Rj = std::max(Rj, Cj.width(d));
    }

    Ri = (Ri / 2 * 1.00001) / theta;
    Rj = (Rj / 2 * 1.00001) / theta;

    if (R2 > (Ri + Rj) * (Ri + Rj)) {                   // If distance is far enough
      M2L(Ci, Cj, Xperiodic);                           //  M2L kernel
    } else if (Ci.IsLeaf() && Cj.IsLeaf()) {            // Else if both cells are bodies
      TapasFMM::Map(P2P_Kernel(false), tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic);
    } else {                                            // Else if cells are close but not bodies
      tapas_splitCell(Ci, Cj, Ri, Rj, theta);           //  Split cell and call function recursively for child
    }                                                   // End if for multipole acceptance
  }

  template<class Cell>
  inline void tapas_splitCell(Cell &Ci, _CONST Cell &Cj, real_t Ri, real_t Rj, real_t theta) {
    (void) Ri; (void) Rj;
    
    if (Cj.IsLeaf()) {
      assert(!Ci.IsLeaf());                                   //  Make sure Ci is not leaf
      TapasFMM::Map(*this, tapas::Product(Ci.subcells(), Cj), theta);
    } else if (Ci.IsLeaf()) {                                   // Else if Ci is leaf
      assert(!Cj.IsLeaf());                                   //  Make sure Cj is not leaf
      TapasFMM::Map(*this, tapas::Product(Ci, Cj.subcells()), theta);
    } else if (Ci == Cj) {
      TapasFMM::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), theta);
#if 1
    } else if (Ri >= Rj) {
      // 1-side split
      TapasFMM::Map(*this, tapas::Product(Ci.subcells(), Cj), theta);
    } else {                                                    // Else if Cj is larger than Ci
      TapasFMM::Map(*this, tapas::Product(Ci, Cj.subcells()), theta);
    }
#else
    // } else {
    //   // 2-side split
    //   TapasFMM::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), theta);
    // }
#endif
  }
};

void CheckResult(Bodies &bodies, int numSamples, real_t cycle, int images) {
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  numSamples = std::min(numSamples, (int)bodies.size());

  Bodies targets(numSamples);
  Bodies samples(numSamples);

  int stride = bodies.size() / numSamples;

  //std::cout << "numSamples = " << numSamples << std::endl;
  //std::cout << "stride=" << stride << std::endl;
  //std::cout << "bodies.size() = " << bodies.size() << std::endl;

  if (mpi_rank == 0) {
    for (int i=0, j=0; i < numSamples; i++,j+=stride) {
      samples[i] = bodies[j];
      targets[i] = bodies[j];
    }
    Dataset().initTarget(samples);
  }

  int prange = 0;
  for (int i=0; i<images; i++) {
    prange += int(std::pow(3.,i));
  }

  for (int p = 0; p < mpi_size; p++) {
    if (p == mpi_rank) {
      std::cout << "Computing on rank " << p << " against " << bodies.size() << " bodies." << std::endl;

      vec3 Xperiodic = 0;
      for (int ix=-prange; ix<=prange; ix++) {
        for (int iy=-prange; iy<=prange; iy++) {
          for (int iz=-prange; iz<=prange; iz++) {
            Xperiodic[0] = ix * cycle;
            Xperiodic[1] = iy * cycle;
            Xperiodic[2] = iz * cycle;

            for (size_t i = 0; i < samples.size(); i++) {
              for (size_t j = 0; j < bodies.size(); j++) {
                // By passing true to thte ctor of P2P class,
                // debug print in P2P::operator() is enabled. See LaplhaseP2PCPU_tapas.cxx
                P2P(false)(samples[i], samples[i].TRG, bodies[j], bodies[j].TRG, Xperiodic);
                //P2P(false)(Ci->BODY[i], Ci->BODY[i].TRG, Cj->BODY[j], Cj->BODY[j].TRG, Xperiodic);
              }
            }
          }
        }
      }
    }

#ifdef USE_MPI
    // Send sampled bodies to rank p to rank (p+1) % mpi_size
    int src = p;
    int dst = (p + 1) % mpi_size;

    if (src != dst) {
      if (src == mpi_rank) {
        MPI_Send(samples.data(), sizeof(samples[0]) * samples.size(), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
      } else if (dst == mpi_rank) {
        MPI_Status stat;
        MPI_Recv(samples.data(), sizeof(samples[0]) * samples.size(), MPI_BYTE, src, 0, MPI_COMM_WORLD, &stat);
      }
    }
#endif
  }

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  // Traversal::normalize()
  for (auto b = samples.begin(); b != samples.end(); b++) {
    b->TRG /= b->SRC;
  }

  Verify verify;
  double potDif = verify.getDifScalar(samples, targets);
  double potNrm = verify.getNrmScalar(samples);
  double accDif = verify.getDifVector(samples, targets);
  double accNrm = verify.getNrmVector(samples);

  logger::printTitle("FMM vs. direct");
  // std::cout << "potDif = " << potDif << std::endl;
  // std::cout << "potNrm = " << potDif << std::endl;
  // std::cout << "accDif = " << potDif << std::endl;
  // std::cout << "accNrm = " << potDif << std::endl;
  verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
  verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
  std::cout.flush();
}

/**
 * \brief Copy particle informations from Tapas to user's program to check result
 */
static inline void CopyBackResult(Bodies &bodies, TapasFMM::Cell *root) {
  bodies.clear();

  Body *beg = &root->local_body(0);
  Body *end = beg + root->local_nb();
  bodies.assign(beg, end); // assign body attributes

  for (size_t i = 0; i < bodies.size(); i++) {
    bodies[i].TRG = root->local_body_attr(i);
  }
}

std::string Now() {
  time_t now = time(NULL);
  struct tm *pnow = localtime(&now);

  std::stringstream ss;
  ss << pnow->tm_year + 1900 << "/"
     << (pnow->tm_mon + 1) << "/"
     << pnow->tm_mday << " "
     << pnow->tm_hour << ":"
     << pnow->tm_min << ":"
     << pnow->tm_sec;
  return ss.str();
}

void PrintProcInfo() {
  const constexpr int HOSTNAME_LEN = 50;
  char hostname[HOSTNAME_LEN] = {0};
  gethostname(hostname, HOSTNAME_LEN);

  int pid = getpid();

#ifdef USE_MPI
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char *rbuf_hn = (rank == 0) ? new char[HOSTNAME_LEN * size] : nullptr;
  MPI_Gather(hostname, HOSTNAME_LEN, MPI_BYTE, rbuf_hn, HOSTNAME_LEN, MPI_BYTE, 0, MPI_COMM_WORLD);

  int *rbuf_pid = (rank == 0) ? new int[size] : nullptr;
  MPI_Gather(&pid, 1, MPI_INT, rbuf_pid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      std::string hn(rbuf_hn + HOSTNAME_LEN * i);
      std::cout << "MPI Rank " << i << " " << hn << " " << rbuf_pid[i] << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
#else
  std::cout << "MPI Rank 0" << hostname << ":" << pid << std::endl;
#endif
}

int main(int argc, char ** argv) {
  Args args(argc, argv);

#ifdef USE_MPI
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &args.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &args.mpi_size);
#endif

  PrintProcInfo();

#ifdef TBB
  if (TBB_INTERFACE_VERSION != TBB_runtime_interface_version()) {
    if (args.mpi_rank == 0) {
      std::cerr << "Compile-time and run-time TBB versions do not match." << std::endl;
    }
    abort();
  }

  if (args.mpi_rank == 0) {
    std::cout << "TBB: version = " << TBB_runtime_interface_version() << std::endl;
    std::cout << "TBB: Initializing threads = " << args.threads << std::endl;
  }
  task_scheduler_init init(args.threads);
#endif

  if (args.mpi_rank == 0) {
#ifdef TAPAS_USE_WEIGHT
    std::cout << "Weighted re-partitioning is activated." << std::endl;
#else
    std::cout << "Weighted re-partitioning is NOT activated." << std::endl;
#endif
  }

  // This function is called by Tapas automatically and user program doesn't need to call it actually.
  // In this program, however, we want to exclude initialization time of CUDA runtime from performance
  // measurement.
  tapas::SetGPU();

  if (args.mpi_rank == 0) {
    std::cout << "Threading model " << TapasFMM::Threading::name() << std::endl;
  }

  // ad-hoc code for MassiveThreads when used with mvapich.
  TapasFMM::Threading::init();

  Bodies bodies;
  //Cells cells, jcells;
  Dataset data;

  if (args.useRmax) {
    std::cerr << "Rmax not supported." << std::endl;
    std::cerr << "Use --useRmax 0 option." << std::endl;
    exit(1);
  }
  if (args.useRopt) {
    std::cerr << "Ropt not supported." << std::endl;
    std::cerr << "Use --useRopt 0 option." << std::endl;
    exit(1);
  }

  //UpDownPass upDownPass(args.theta, args.useRmax, args.useRopt);
  Verify verify;
  (void) verify;

  Region tr;

  logger::startTimer("Dataset generation");
  bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
  logger::stopTimer("Dataset generation");

  // Dump all bodies data for debugging
#ifdef TAPAS_DEBUG_DUMP
  {
    tapas::debug::DebugStream err("bodies");
    for (auto &b : bodies) {
      err.out() << b.X << " " << b.SRC  << " " << b.TRG << std::endl;
    }
  }
#endif

  const real_t cycle = 2 * M_PI;
  logger::verbose = args.verbose && (args.mpi_rank == 0);
  if (args.mpi_rank == 0) {
    logger::printTitle("FMM Parameters");
    args.print(logger::stringLength, P);
  }

  if (args.mpi_rank == 0) {
    std::cout << "Starting FMM timesteps" << std::endl;
  }

#ifdef USE_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  TapasFMM::Cell *root = nullptr;

  // Start timesteps
  for (int t = 0; t < args.repeat; t++) {
    if (rank == 0) {
      std::cout << "===== Timestep " << t << " =====" << std::endl;
    }
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();

    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Partition");
      if (root == nullptr) {
        // first timestep
        root = TapasFMM::Partition(bodies.data(), bodies.size(),
                                   args.ncrit, MPI_COMM_WORLD);
      } else {
        // otherwise
        root = TapasFMM::Partition(root, args.ncrit);
      }
      logger::stopTimer("Partition");
    }

    // Upward (P2M + M2M)
    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Upward pass");

      if (!root->IsLeaf()) {
        TapasFMM::Map(FMM_Upward(), root->subcells(), args.theta);
      }
      
      logger::stopTimer("Upward pass");
    }

#ifdef TAPAS_DEBUG_DUMP
    dumpM(*root);
#endif

    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Traverse");
      ResetCount();

      TapasFMM::Map(FMM_DTT(), tapas::Product(*root, *root), args.theta);

      logger::stopTimer("Traverse");
    }

    TAPAS_LOG_DEBUG() << "Dual Tree Traversal done\n";

#ifdef TAPAS_DEBUG_DUMP
    dumpL(*root);
#endif

    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Downward pass");

      if (root->IsLeaf()) {
        TapasFMM::Map(L2P, root->bodies());
      } else {
        TapasFMM::Map(FMM_Downward(), root->subcells());
      }

      logger::stopTimer("Downward pass");
    }

    TAPAS_LOG_DEBUG() << "L2P done\n";

#ifdef TAPAS_DEBUG_DUMP
    dumpBodies(*root);
#endif

    // Copy BodyAttr values back to Body

    logger::startTimer("CopyBackResult");
    TapasFMM::Map([](Body &b, BodyAttr &a) {
        b.TRG = a;
      }, root->Bodies());
    CopyBackResult(bodies, root);
    logger::stopTimer("CopyBackResult");

    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");

#if WRITE_TIME
    logger::writeTime();
#endif

#ifdef COUNT
    if (args.mpi_rank == 0) {
      std::cout << "P2P calls" << " : " << numP2P << std::endl;
      std::cout << "M2L calls" << " : " << numM2L << std::endl;
    }
#endif

    //buildTree.printTreeData(cells);
    logger::printPAPI();
    logger::stopDAG();
    
    root->Report();

    if (t == args.repeat - 1) { // Final Timesteps
      if (args.check) {
        const int numTargets = 10;
        logger::startTimer("Total Direct");
        CheckResult(bodies, numTargets, cycle, args.images);
        logger::stopTimer("Total Direct");
      }
      TapasFMM::Destroy(root);
    } else {
      // Prepare for the next timestep
      data.initTarget(bodies);
    }

    logger::timer.clear();
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  } /* end for t */

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
