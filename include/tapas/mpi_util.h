#ifndef TAPAS_MPI_UTIL_
#define TAPAS_MPI_UTIL_

#include <cmath>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <mpi.h>

#include <tapas/common.h>

#include "tapas/debug_util.h"

#ifdef TAPAS_DEBUG
#define MPI_CHECK(ret_, comm_) do {                                 \
    if (ret_ != MPI_SUCCESS) {                                      \
      int rank = -1;                                                \
      char err_str[200];                                            \
      int len;                                                      \
      MPI_Error_string(ret_, err_str, &len);                        \
      MPI_Comm_rank(comm_, &rank);                                  \
      err_str[len] = '\0';                                          \
      fprintf(stderr, "MPI_SAFE_CALL: MPI failed on rank %d: %s\n", \
              rank, err_str);                                       \
      abort();                                                      \
    }                                                               \
  } while(0)
#else
#define MPI_CHECK(ret_, comm_) (void)ret_; (void)comm_;
#endif


namespace tapas {
namespace util {

/**
 * \brief Generic inclusive scan
 *
 * Prototypes of inclusive_scan and exclusive_scan are inspired by
 * "Working Draft, Technical Specification for C++ Extensions for Parallelism"
 * by Jared Hoberock (NVIDIA Corporation)
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4310.html#parallel.alg.inclusive.scan
 */
template<class InputIterator, class OutputIterator,
         class BinaryOperation>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  using value_type = typename InputIterator::value_type;
  value_type tally = *first;
  InputIterator iter = first + 1;
  *result = tally;
  result++;

  while (iter != last) {
    tally = binary_op(tally, *iter);
    *result = tally;
    result++;
    iter++;
  }

  return result;
}

template<class InputIterator, class OutputIterator>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  return inclusive_scan(first, last, result,
                        std::plus<typename InputIterator::value_type>());
}


template<class InputIterator, class OutputIterator,
         class BinaryOperation>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  using value_type = typename InputIterator::value_type;
  value_type tally = value_type();
  InputIterator iter = first;

  *result = tally;
  result++;

  while (iter + 1 != last) {
    tally = binary_op(tally, *iter);
    *result = tally;
    result++;
    iter++;
  }

  return result;
}


template<class InputIterator, class OutputIterator>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  return exclusive_scan(first, last, result,
                        std::plus<typename InputIterator::value_type>());
}



} // namespace util
} // namespace tapas


namespace tapas {
namespace mpi {

/**
 * \brief Returns MPI rank in the specified communicator
 */
int Rank(MPI_Comm comm = MPI_COMM_WORLD) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

/**
 * \brief Returns MPI size in the specified communicator
 */
int Size(MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

template <class T> void *void_cast(T* p) { return reinterpret_cast<void*>(p); }
template <class T> void *void_cast(const T* p) { return const_cast<void*>(reinterpret_cast<const void*>(p)); }

using tapas::util::exclusive_scan;

template<class T>
void* mpi_sendbuf_cast(const T* p) {
  return const_cast<void*>(reinterpret_cast<const void*>(p));
}

template<class T>
void* mpi_sendbuf_cast(T* p) {
  return reinterpret_cast<void*>(p);
}

// MPI-related utilities and wrappers
// TODO: wrap them as a pluggable policy/traits class
template<class T> struct MPI_DatatypeTraits {
  static MPI_Datatype type() {
    return MPI_BYTE;
  }
  static constexpr bool IsEmbType() {
    return false;
  }

  static constexpr int count(size_t n) {
    return sizeof(T) * n;
  }
};

#define DEF_MPI_DATATYPE(ctype_, mpitype_)        \
  template<> struct MPI_DatatypeTraits<ctype_>  { \
    static MPI_Datatype type() {                  \
      return mpitype_;                            \
    }                                             \
    static constexpr bool IsEmbType() {           \
      return true;                                \
    }                                             \
    static constexpr int count(size_t n) {        \
      return n;                                   \
    }                                             \
  }

DEF_MPI_DATATYPE(char, MPI_CHAR);
DEF_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
DEF_MPI_DATATYPE(wchar_t, MPI_WCHAR);

DEF_MPI_DATATYPE(short, MPI_SHORT);
DEF_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

DEF_MPI_DATATYPE(int, MPI_INT);
DEF_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

DEF_MPI_DATATYPE(long, MPI_LONG);
DEF_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

DEF_MPI_DATATYPE(long long, MPI_LONG_LONG);
DEF_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

DEF_MPI_DATATYPE(float,  MPI_FLOAT);
DEF_MPI_DATATYPE(double, MPI_DOUBLE);
DEF_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

// MPI::COMPLEX Complex<float>
// MPI::DOUBLE_COMPLEX Complex<double>
// MPI::LONG_DOUBLE_COMPLEX Complex<long double>
// MPI::BYTE

template<typename T>
void Allreduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
  auto kType = MPI_DatatypeTraits<T>::type();

  if (!MPI_DatatypeTraits<T>::IsEmbType()) {
    TAPAS_ASSERT(0 && "Allreduce() is not supported for user-defined types.");
  }

  int ret = MPI_Allreduce(mpi_sendbuf_cast(sendbuf), (void*)recvbuf, count, kType, op, comm);
  MPI_CHECK(ret, comm);
}

template<typename T>
void Allreduce(const std::vector<T>& sendbuf, std::vector<T> &recvbuf,
               MPI_Op op, MPI_Comm comm) {
  size_t len = sendbuf.size();
  recvbuf.resize(len);

  Allreduce(sendbuf.data(), recvbuf.data(), (int) len, op, comm);
}

template<typename T>
void Allreduce(T sendval, T &recvval, MPI_Op op, MPI_Comm comm) {
  Allreduce(&sendval, &recvval, 1, op, comm);
}

template<typename T>
void Alltoall(const T *sendbuf, T *recvbuf, int count, MPI_Comm comm) {
  const auto kType = MPI_DatatypeTraits<T>::type();
  int size = MPI_DatatypeTraits<T>::IsEmbType() ? count : count * sizeof(T);
  int ret = ::MPI_Alltoall(sendbuf, size, kType,
                           recvbuf, size, kType,
                           comm);
  (void)ret; // to avoid warnings of 'unused variable'
  TAPAS_ASSERT(ret == MPI_SUCCESS);
}

template<typename T>
void Alltoall(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int count, MPI_Comm comm) {
  recvbuf.resize(sendbuf.size());
  Alltoall(sendbuf.data(), recvbuf.data(), count, comm);
}

template<typename T>
void Reduce(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, MPI_Op op, int root, MPI_Comm comm) {
  recvbuf.resize(sendbuf.size());
  MPI_Reduce(void_cast(sendbuf.data()), void_cast(recvbuf.data()), sendbuf.size(),
             MPI_DatatypeTraits<T>::type(), op, root, comm);
}

template<typename T>
void Reduce(const T& sendval, T& recvval, MPI_Op op, int root, MPI_Comm comm) {
  MPI_Reduce(void_cast(&sendval), void_cast(&recvval), 1,
             MPI_DatatypeTraits<T>::type(), op, root, comm);
}

/**
 * \brief Perform MPI_Alltoallv
 * \tparam T data type
 * \param send_buf Data to be sent
 * \param dest Destination process number of each element of send_buf (i.e. send_buf[i] is sent to dest[i])
 * \param recv_buf (Output parameter) received data
 * \param src (Output parameter) source process number of each element of recv_buf (i.e. recv_buf[i] is from src[i])
 *
 * Caution: send_buf and dest will be sorted in-place.
 */
template<typename VectorType>
void Alltoallv2(VectorType& send_buf, std::vector<int>& dest,
                VectorType& recv_buf, std::vector<int>& src,
                MPI_Datatype dtype, MPI_Comm comm) {
#ifdef TAPAS_DEBUG
  {
    // Check type and VectorType::value_type are consistent
    MPI_Aint extent;
    MPI_Type_extent(dtype, &extent);
    TAPAS_ASSERT(extent == sizeof(typename VectorType::value_type));
  }
#endif

  int mpi_size;

  MPI_Comm_size(comm, &mpi_size);

  TAPAS_ASSERT(send_buf.size() == dest.size());
  SortByKeys(dest, send_buf);

  std::vector<int> send_counts(mpi_size);
  for(int p = 0; p < mpi_size; p++) {
    auto range = std::equal_range(dest.begin(), dest.end(), p);
    send_counts[p] = range.second - range.first;
    TAPAS_ASSERT(send_counts[p] >= 0); // check overflow
  }

  std::vector<int> recv_counts(mpi_size);

  int err = MPI_Alltoall((void*)send_counts.data(), 1, MPI_INT,
                         (void*)recv_counts.data(), 1, MPI_INT,
                         comm);

  if (err != MPI_SUCCESS) {
    TAPAS_ASSERT(!"MPI_Alltoall failed.");
  }

  std::vector<int> send_disp(mpi_size, 0); // displacement
  std::vector<int> recv_disp(mpi_size, 0);

  // exclusive scan
  for (int p = 1; p < mpi_size; p++) {
    send_disp[p] = send_disp[p-1] + send_counts[p-1];
    recv_disp[p] = recv_disp[p-1] + recv_counts[p-1];

    TAPAS_ASSERT(send_disp[p] >= 0); // check overflow
    TAPAS_ASSERT(recv_disp[p] >= 0); // check overflow
  }

  int total_recv_counts = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

  recv_buf.resize(total_recv_counts);
  TAPAS_ASSERT(send_disp.size() == recv_disp.size());

#ifdef TAPAS_DEBUG_COMM_MATRIX
  // debug: print send matrix
  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f (std::cout.flags());

      if (rank == 0) { std::cout << "Comm matrix" << std::endl; }
      std::cout << std::right << std::fixed << std::setw(3) << rank << " ";
      long total = 0;
      for (size_t i = 0; i < send_disp.size(); i++) {
        std::cout << std::right << std::setw(10) << std::fixed << send_counts[i] << " ";
        total += send_counts[i];
      }
      std::cout << "total: " << std::right << std::setw(10) << total;
      std::cout << std::endl;

      std::cout.flags(f);
    });

  // debug: print send displacement
  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f (std::cout.flags());
      if (rank == 0) { std::cout << "Send disp" << std::endl; }
      std::cout << std::right << std::fixed << std::setw(3) << rank << " ";
      for (size_t i = 0; i < send_disp.size(); i++) {
        std::cout << std::right << std::setw(10) << std::fixed << send_disp[i] << " ";
      }
      std::cout << std::endl;
      std::cout.flags(f);
    });

  // debug: print recv matrix
  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f (std::cout.flags());
      if (rank == 0) { std::cout << "Comm matrix" << std::endl; }
      std::cout << std::right << std::fixed << std::setw(3) << rank << " ";
      long total = 0;
      for (size_t i = 0; i < recv_disp.size(); i++) {
        std::cout << std::right << std::setw(10) << std::fixed << recv_counts[i] << " ";
        total += recv_counts[i];
      }
      std::cout << "total: " << std::right << std::setw(10) << total;
      std::cout << std::endl;
      std::cout.flags(f);
    });

  // debug: print recv displacement
  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f (std::cout.flags());
      if (rank == 0) { std::cout << "Recv disp" << std::endl; }
      std::cout << std::right << std::fixed << std::setw(3) << rank << " ";
      for (size_t i = 0; i < recv_disp.size(); i++) {
        std::cout << std::right << std::setw(10) << std::fixed << recv_disp[i] << " ";
      }
      std::cout << std::endl;
      std::cout.flags(f);
    });

  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f (std::cout.flags());
      if (rank == 0) {
        std::cout << "MPI_Alltoallv() Starting." << std::endl;
      }

      if (rank == 2) {
        std::cout << "send_buf.size() = " << send_buf.size() << std::endl;
        size_t size = send_buf.size() * sizeof(send_buf[0]);
        std::cout << "total size of send_buf = " << size << " (=" << (size/1024./1024/1024) << " GB)" << std::endl;
        std::cout << "int max = " << std::numeric_limits<int>::max() << std::endl;
        std::cout << std::endl;
        std::cout << "recv_buf.size() = " << recv_buf.size() << std::endl;
      }
      std::cout.flags(f);
    });
#endif

  int ret = MPI_Alltoallv((void*)send_buf.data(), send_counts.data(), send_disp.data(), dtype,
                          (void*)recv_buf.data(), recv_counts.data(), recv_disp.data(), dtype,
                          comm);
  MPI_CHECK(ret, comm);

#ifdef TAPAS_DEBUG_COMM_MATRIX
  tapas::debug::BarrierExec([](int rank, int) {
      if (rank == 0) {
        std::cout << "MPI_Alltoallv() done." << std::endl;
      }
    });
#endif

  // Build src[] array
  src.clear();
  src.resize(total_recv_counts, 0);

  {
    int p = 0;
    for (int i = 0; i < total_recv_counts; i++) {
      while (p < mpi_size-1 && i >= recv_disp[p+1]) {
        p++;
      }
      src[i] = p;
    }
  }

  auto src2 = src;
  src2.clear(); src2.resize(src.size(), 0);

  index_t pos = 0;
  for (size_t p = 0; p < recv_counts.size(); p++) {
    int num_recv_from_p = recv_counts[p];

    for (int i = 0; i < num_recv_from_p; i++, pos++) {
      src2[pos] = p;
    }
  }

#if 1
  // TODO: bug? May be src2 is the correct answer?
  src = src2;
#endif
}

template<class T>
void Alltoallv(const std::vector<T> &send_buf,
               const std::vector<int> &send_count,
               std::vector<T> &recv_buf, std::vector<int> &recv_count,
               MPI_Comm comm) {
  int mpi_size;

  MPI_Comm_size(comm, &mpi_size);

  recv_count.clear();
  recv_count.resize(mpi_size);

  int err = MPI_Alltoall(void_cast(send_count.data()), 1, MPI_INT,
                         void_cast(recv_count.data()), 1, MPI_INT,
                         comm);

  if (err != MPI_SUCCESS) {
    TAPAS_ASSERT(!"MPI_Alltoall failed.");
  }

  std::vector<int> send_disp(mpi_size, 0); // displacement
  std::vector<int> recv_disp(mpi_size, 0);

  // exclusive scan
  for (int p = 1; p < mpi_size; p++) {
    send_disp[p] = send_disp[p-1] + send_count[p-1];

    recv_disp[p] = recv_disp[p-1] + recv_count[p-1];
  }

  int total_recv_count = std::accumulate(recv_count.begin(), recv_count.end(), 0);

  recv_buf.resize(total_recv_count);

  auto kType = MPI_DatatypeTraits<T>::type();

#ifdef TAPAS_DEBUG_COMM_MATRIX
  // Print a communication matrix
  tapas::debug::BarrierExec([&](int rank, int) {
      std::ios::fmtflags f(std::cout.flags());

      if (rank == 0) {
        std::cout << "MPI_Alltoallv comm. matrix" << std::endl;
        std::cout << " ";
      }

      std::cout << std::setw(3) << rank << " ";
      for (int p = 0; p < mpi_size; p++) {
        std::cout << std::right << std::setw(10) << std::fixed << send_count[p] << " ";
      }

      std::cout << std::endl;
      std::cout.flags(f);
    });
#endif

  auto send_count2 = send_count;
  auto send_disp2 = send_disp;
  auto recv_count2 = recv_count;
  auto recv_disp2 = recv_disp;

  if(!MPI_DatatypeTraits<T>::IsEmbType()) {
    kType = MPI_BYTE;
    // If T is not an embedded type, MPI_BYTE and sizeof(T) is used.
    for (size_t i = 0; i < recv_disp.size(); i++) {
      recv_disp2[i] = recv_disp[i] * sizeof(T);
      recv_count2[i] = recv_count[i] * sizeof(T);
    }
    for (size_t i = 0; i < send_disp.size(); i++) {
      send_disp2[i] = send_disp[i] * sizeof(T);
      send_count2[i] = send_count[i] * sizeof(T);
    }
  }

  int ret = MPI_Alltoallv(void_cast(send_buf.data()), send_count2.data(), send_disp2.data(), kType,
                          void_cast(recv_buf.data()), recv_count2.data(), recv_disp2.data(), kType,
                          comm);
  MPI_CHECK(ret, comm);
}

template<class T>
void Gather(const T& val, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto type = MPI_DatatypeTraits<T>::type();
  int count = MPI_DatatypeTraits<T>::count(1);

  if (rank == root) {
    recvbuf.clear();
    recvbuf.resize(size);
  } else {
    recvbuf.clear();
  }

  int ret = ::MPI_Gather(void_cast(&val), count, type,
                         void_cast(&recvbuf[0]), count, type, root, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Gather(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  //MPI_Comm_size(comm, &size);

  auto type = MPI_DatatypeTraits<T>::type();
  int count = MPI_DatatypeTraits<T>::count(sendbuf.size());

  if (rank == root) {
    recvbuf.clear();
    recvbuf.resize(sendbuf.size() * size);
  } else {
    recvbuf.clear();
  }

  int ret = ::MPI_Gather(void_cast(&sendbuf[0]), count, type,
                         void_cast(&recvbuf[0]), count, type, root, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Scatter(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == root) {
    assert((int)sendbuf.size() % size == 0);
  }

  int count = sendbuf.size() / size;

  recvbuf.clear();
  recvbuf.resize(count);

  auto type = MPI_DatatypeTraits<T>::type();
  int sendcount = MPI_DatatypeTraits<T>::count(count);

  int ret = ::MPI_Scatter(void_cast(sendbuf.data()), sendcount, type,
                          void_cast(recvbuf.data()), sendcount, type, root, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Scatter(const std::vector<T> &sendbuf, T &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == root) {
    assert((int)sendbuf.size() == size);
  }

  auto type = MPI_DatatypeTraits<T>::type();
  int sendcount = MPI_DatatypeTraits<T>::count(1);

  int ret = ::MPI_Scatter(void_cast(sendbuf.data()), sendcount, type,
                          void_cast(&recvbuf), sendcount, type, root, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Gatherv(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<int> recvcounts; // only significant at root
  std::vector<int> displs; // only significant at root

  auto type = MPI_DatatypeTraits<T>::type();
  int count = MPI_DatatypeTraits<T>::count(sendbuf.size());

  // Gather recvcounts to root process
  Gather(count, recvcounts, root, comm);

  if (rank == root) {
    assert(recvcounts.size() == (size_t)size);
    exclusive_scan(recvcounts.begin(), recvcounts.end(), back_inserter(displs));
    int total_recv_count = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    
    recvbuf.clear();
    if (MPI_DatatypeTraits<T>::IsEmbType()) {
      recvbuf.resize(total_recv_count);
    } else {
      recvbuf.resize(total_recv_count / sizeof(T));
    }
  } else {
    recvbuf.clear();
  }
  
  int ret = ::MPI_Gatherv(void_cast(sendbuf.data()), count, type,
                          void_cast(recvbuf.data()), recvcounts.data(), displs.data(), type, root, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Allgather(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, MPI_Comm comm) {
  int mpi_size = tapas::mpi::Size();
  
  auto kType = MPI_DatatypeTraits<T>::type();
  int count = sendbuf.size();
  if (kType == MPI_BYTE) {
    count *= sizeof(T);
  }

  recvbuf.clear();
  recvbuf.resize(count * mpi_size);

  int ret = ::MPI_Allgather(mpi_sendbuf_cast(sendbuf.data()), count, kType, 
                            recvbuf.data(), count, kType, comm);

  TAPAS_ASSERT(ret == MPI_SUCCESS);
}


template<class T>
void Allgatherv(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, MPI_Comm comm) {
  int size = -1, rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int count = sendbuf.size();
  std::vector<int> recvcounts(size);

  auto kType = MPI_DatatypeTraits<T>::type();

  // Call allgather and create recvcount & displacements array.
  int ret = ::MPI_Allgather(mpi_sendbuf_cast(&count), 1, MPI_INT,
                            reinterpret_cast<void*>(recvcounts.data()), 1, MPI_INT, comm);
  (void)ret;
  TAPAS_ASSERT(ret == MPI_SUCCESS);

  std::vector<int> disp;
  exclusive_scan(recvcounts.begin(), recvcounts.end(), back_inserter(disp));

  int recvcount = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
  recvbuf.resize(recvcount);

  if (kType == MPI_BYTE) {
    count *= sizeof(T);
    recvcount *= sizeof(T);
    for (auto && c : recvcounts) c *= sizeof(T);
    for (auto && d : disp) d *= sizeof(T);
  }

  ret = ::MPI_Allgatherv(mpi_sendbuf_cast(sendbuf.data()), count, kType,
                         reinterpret_cast<void*>(recvbuf.data()), recvcounts.data(), disp.data(),
                         kType, comm);


  (void)ret;
  TAPAS_ASSERT(ret == MPI_SUCCESS);
}

template<class T>
void Bcast(std::vector<T> &buf, int root, MPI_Comm comm) {
  int rank = 0;

  MPI_Comm_rank(comm, &rank);

  ::MPI_Bcast(reinterpret_cast<void*>(buf.data()),
              MPI_DatatypeTraits<T>::count(buf.size()),
              MPI_DatatypeTraits<T>::type(),
              root,
              comm);
}


} // namespace mpi
} // namespace tapas

#endif // TAPAS_MPI_UTIL_
