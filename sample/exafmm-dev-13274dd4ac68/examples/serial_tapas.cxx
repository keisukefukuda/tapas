#include <unistd.h> // usleep

#ifdef EXAFMM_TAPAS_MPI
# include <mpi.h>
#endif

#include "args.h"
#include "bound_box.h"
#ifdef CILK
#include "build_tree3.h"
#else
#include "build_tree.h"
#endif
#include "dataset.h"
#include "logger.h"
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"

#define TAPAS

#ifdef TAPAS
#include "tapas_exafmm.h"
#include "serial_tapas_helper.cxx"
#endif

int main(int argc, char ** argv) {
  Args args(argc, argv);
  
#ifdef EXAFMM_TAPAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &args.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &args.mpi_size);
#endif
    
  Bodies bodies, bodies2, bodies3, jbodies;
  BoundBox boundBox(args.nspawn);
  Bounds bounds;
  BuildTree buildTree(args.ncrit, args.nspawn);
  Cells cells, jcells;
  Dataset data;
  Traversal traversal(args.nspawn, args.images);

  
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
  UpDownPass upDownPass(args.theta, args.useRmax, args.useRopt);
  Verify verify;

  Region tr;

  num_threads(args.threads);

#ifdef EXAFMM_TAPAS_MPI
  for (int i = 0; i < args.mpi_size; i++) {
      if (args.mpi_rank == i) {
          bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
      }
      MPI_Barrier(MPI_COMM_WORLD);
  }
#else
  bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
#endif

#ifdef EXAFMM_TAPAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < args.mpi_size; i++) {
      if (args.mpi_rank == i) {
          bool append_mode = (i > 0); // Previously existing file is truncated by rank 0
          data.DumpToFile(bodies, "init_bodies.dat", append_mode);
      }
      usleep(10000);
      MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  if (args.mpi_rank == 0) {
      data.DumpToFile(bodies, "init_bodies.dat");
  }
#endif

  const real_t cycle = 2 * M_PI;
  logger::verbose = args.verbose && (args.mpi_rank == 0);
  if (args.mpi_rank == 0) {
      logger::printTitle("FMM Parameters");
      args.print(logger::stringLength, P);
  }

  for (int t=0; t<args.repeat; t++) {
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();
    bounds = boundBox.getBounds(bodies);
#ifdef TAPAS
    asn(tr, bounds);
    TAPAS_LOG_DEBUG() << "Bounding box: " << tr << std::endl;
#endif

    Tapas::Cell *root = Tapas::Partition(
        bodies.data(), args.numBodies, tr, args.ncrit);
#if 0
    {
      std::ofstream tapas_out("tapas_0.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif
    
    logger::startTimer("Upward pass");                          // Start timer
    tapas::Map(FMM_P2M, *root, args.theta);
    logger::stopTimer("Upward pass");                          // Start timer
    TAPAS_LOG_DEBUG() << "P2M done\n";
#if 0
    {
      std::ofstream tapas_out("tapas_P2M.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif


    logger::startTimer("Traverse");
    numM2L = 0; numP2P = 0;
    tapas::Map(FMM_M2L, tapas::Product(*root, *root), args.mutual, args.nspawn);
    logger::stopTimer("Traverse");
    
    TAPAS_LOG_DEBUG() << "M2L done\n";
    jbodies = bodies;
#if 0
    {
      std::ofstream tapas_out("tapas_M2L.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif

    logger::startTimer("Downward pass");
    tapas::Map(FMM_L2P, *root);
    logger::stopTimer("Downward pass");
    
    TAPAS_LOG_DEBUG() << "L2P done\n";
#if 0
    {
      std::ofstream tapas_out("tapas_L2P.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif


    CopyBackResult(bodies, root->body_attrs(), args.numBodies);
#if 0
    {
      std::ofstream tapas_out("tapas_final.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << bodies[i].TRG << std::endl;
        //tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif

    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");
#if WRITE_TIME
    logger::writeTime();
#endif
    const int numTargets = 100;
    bodies3 = bodies;
    data.sampleBodies(bodies, numTargets);
    bodies2 = bodies;
    data.initTarget(bodies);
    logger::startTimer("Total Direct");
    traversal.direct(bodies, jbodies, cycle);
    traversal.normalize(bodies);
    logger::stopTimer("Total Direct");
    double potDif = verify.getDifScalar(bodies, bodies2);
    double potNrm = verify.getNrmScalar(bodies);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);
    logger::printTitle("FMM vs. direct");
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
    buildTree.printTreeData(cells);
    traversal.printTraversalData();
    logger::printPAPI();
    logger::stopDAG();
    bodies = bodies3;
    data.initTarget(bodies);
  }
  logger::writeDAG();
#ifdef EXAFMM_TAPAS_MPI
  MPI_Finalize();
#endif
  return 0;
}
