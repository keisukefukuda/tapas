#ifndef TAPAS_HOT_INSPECTOR_REGION_DEPTH_BB_MAP_H_
#define TAPAS_HOT_INSPECTOR_REGION_DEPTH_BB_MAP_H_

#include <mpi.h>

namespace tapas {
namespace hot {

namespace {
}

// Depth Map :: (rank, depth) -> Region
template<class CELL>
class DepthBBMap {
  using CellType = CELL;
  using KeyType = typename CELL::KeyType;
  using CellHashTable = std::unordered_map<KeyType, CellType*>;
  using Reg = typename CELL::Reg;

  using DepthToBB = std::unordered_map<int, Reg>;
  using RankToMap = std::unordered_map<int, DepthToBB>;
  
  const int mpi_size_;
  const int mpi_rank_;
  const MPI_Comm mpi_comm_;
  const int max_depth_;
  RankToMap map_;

 public:
  DepthBBMap(int max_depth, const CellHashTable& hash, MPI_Comm comm)
      : mpi_size_(tapas::mpi::Size(comm))
      , mpi_rank_(tapas::mpi::Rank(comm))
      , mpi_comm_(comm)
      , max_depth_(max_depth)
      , map_()
  {
    
    // First, build a depth BB map for the current process.
    DepthToBB m;
    int dmax = -1;
    
    for (auto &&entry : hash) {
      const CellType &c = *entry.second;
      int d = c.depth();

      if (m.count(d) == 0) {
        m[d] = c.GetRegion();
      } else {
        m[d] = Reg::BB(m[d], c.GetRegion());
      }

      dmax = std::max(dmax, d);
    }

    for (int d = dmax + 1; d <= max_depth_; d++) {
      m[d] = m[dmax];
    }

    map_[mpi_rank_] = m;
  }

  inline const Reg &operator()(int depth) const {
    return (*this)(mpi_rank_, depth);
  }

  inline const Reg &operator()(int rank, int depth) const {
    TAPAS_ASSERT(rank < mpi_size_);
    TAPAS_ASSERT(map_.count(rank) == 1);

    return map_.at(rank).at(depth);
  }

  void Exchange() {
    std::vector<Reg> send_buf(max_depth_);

    for (int dep = 0; dep < max_depth_; dep++) {
      send_buf[dep] = map_.at(mpi_rank_).at(dep);
    }

    std::vector<Reg> recv_buf;
    tapas::mpi::Allgather(send_buf, recv_buf, mpi_comm_);

    for (int rank = 0; rank < mpi_size_; rank++) {
      if (rank == mpi_rank_) continue;

      DepthToBB m;
      int idx = max_depth_ * rank;
      for (int dep = 0; dep < max_depth_; dep++) {
        m[dep] = recv_buf[idx + dep];
      }
      map_[rank] = m;
    }
  }
};


}
}

#endif // TAPAS_HOT_INSPECTOR_REGION_DEPTH_BB_MAP_H_
