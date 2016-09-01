#ifndef TAPAS_HOT_REPORT_H
#define TAPAS_HOT_REPORT_H

#include <sys/unistd.h> // gethostname

#include <ostream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "tapas/hot.h"
#include "tapas/util.h"

namespace tapas {
namespace hot {

template<class Data>
void PrintParams(std::ofstream &ofs, const Data &data) {
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  char hostname [201];
  gethostname(hostname, 200);
  
  /*
   * 必要な情報：
   * sim: nb, ncrit, dist, sampling rate, 
   * env: host (rank 0), date, time, 
   * 
   */
  ofs << "# MPI size " << data.mpi_size_ << std::endl;
  ofs << "# numBodies " << data.nb_total << std::endl;
  ofs << "# ncrit " << data.ncrit_ << std::endl;
  ofs << "# sampling rate " << data.sample_rate_ << std::endl;
  ofs << "# date " << tm.tm_year + 1900 << "-" << tm.tm_mon + 1 << "-" << tm.tm_mday << " "
      << tm.tm_hour << ":" << tm.tm_min << ":" << tm.tm_sec << std::endl;
  ofs << "# rank 0 on " << hostname << std::endl;
}

template<class Data>
void Report(const Data &data) {
  // auto comm = data.mpi_comm_;
  auto comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string report_fname = "tapas_report";

  if (getenv("TAPAS_REPORT_FILENAME")) {
    report_fname = getenv("TAPAS_REPORT_FILENAME");
  }

  // append .csv extension if necessary
  if (report_fname.find(".csv") == std::string::npos) {
    report_fname = report_fname + ".csv";
  }

  std::ofstream ofs;
  if (rank == 0) {
    tapas::util::OpenFileStream(ofs, report_fname.c_str(), std::ios::out);
    
    // First, print configuration values of the simulation as comments in .csv file.
    PrintParams(ofs, data);
  }

  data.time_rec_.Dump(ofs, rank, size);

#if 0
  if (data.mpi_rank_ == 0) {
    std::cout << "Max Depth = " << data.max_depth_ << std::endl;
  }
#endif
}

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_REPORT_H
