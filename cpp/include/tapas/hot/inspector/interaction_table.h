#ifndef TAPAS_HOT_INSPECTOR_INTERACTION_TABLE_H__
#define TAPAS_HOT_INSPECTOR_INTERACTION_TABLE_H__

#include <vector>

#include <tapas/hot/cell.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/inspector/region_depth_map.h>

namespace tapas {
namespace hot {

template<class TSP, class UserFunct, class...Args>
class InteractionTable {
 public:
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using CellType = Cell<TSP>;
  using KeyType = typename CellType::KeyType;
  using SFC = typename CellType::SFC;
  using Data = typename CellType::Data;
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  using BodyType = typename CellType::BodyType;
  using BodyAttrType = typename CellType::BodyAttrType;

  using CellAttr = typename CellType::CellAttr;
  using VecT = tapas::Vec<TSP::Dim, typename TSP::FP>;
  using Reg = Region<Dim, FP>;

  using TravPolicy = tapas::hot::proxy::OnesideTraversePolicy<Dim, FP, Data>;
  using GCell = tapas::hot::proxy::ProxyCell<TSP, TravPolicy>;
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<GCell>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<GCell>;

 private:
  Data &data_;
  DepthBBMap<CellType> depth2bb_;
  int max_depth_;
  int src_root_depth_;
  int trg_root_depth_;
  int ncol_;
  int nrow_;
  Interaction<TSP, UserFunct, Args...> inter_;
  std::vector<IntrFlag> table_;
  
 public:
  /**
   * \brief Do inspection between a single pair of target (local) root and source (remote) root
   *
   * \param src_key Key of a local rootof the source (remote) process
   * \param trg_key Key of a local root of the target (local) process
   * \param f The user function
   * \param args Arguments to the user function
   */
  InteractionTable(Data &data, KeyType trg_root_key, KeyType src_root_key,
                   UserFunct f, Args... args)
      : data_(data)
      , depth2bb_(data.max_depth_, data.ht_, data.mpi_comm_)
      , max_depth_(data.max_depth_)
      , src_root_depth_(SFC::GetDepth(src_root_key))
      , trg_root_depth_(SFC::GetDepth(trg_root_key))
      , ncol_(max_depth_ - src_root_depth_ + 1)
      , nrow_(max_depth_ - trg_root_depth_ + 1)
      , inter_(data)
      , table_()
  {
    const Reg src_reg = SFC::CalcRegion(src_root_key, data_.region_);
    const Reg trg_reg_root = SFC::CalcRegion(trg_root_key, data_.region_);

    table_.resize(ncol_ * nrow_);

    for (int sd = src_root_depth_; sd <= max_depth_; sd++) { // source depth
      for (int td = trg_root_depth_; td <= max_depth_; td++) { // target depth
        const Reg &trg_reg = depth2bb_(td);

        IntrFlag split;

        split = inter_.RetryIntr(trg_reg, src_reg, td, sd, false, f, args...);
        
        // We here use a ghost cell for the target cell and assume that
        // the target ghost cell is not a leaf.
        // Thus, there is possibility that the source cell is split
        // even if the `split` value above is not SplitR if the target(left) cell is a leaf.
        if (!split.IsSplitR()
            && td >= data_.min_leaf_level_[trg_root_key]) {

          IntrFlag split2 = inter_.RetryIntr(trg_reg, src_reg, td, sd, true, f, args...);

          TAPAS_ASSERT(!split2.IsSplitL()); // because left cell (target) is a leaf.

          if (split2.IsSplitR()) {
            split = IntrFlag(IntrFlag::SplitILL);
          }
        }
        int idx = (td - trg_root_depth_) * ncol_ + (sd - src_root_depth_);
        table_[idx] = split;
      }
    }
  }

  IntrFlag At(int trg_depth, int src_depth) {
    int r = trg_depth - trg_root_depth_; // row
    int c = src_depth - src_root_depth_; // col
    TAPAS_ASSERT(r < nrow_);
    TAPAS_ASSERT(c < ncol_);

    return table_[r * ncol_ + c];
  }

  bool IsRootApprox() const {
    return table_[0].IsApprox();
  }

  IntrFlag TopLeft() const {
    return table_[0];
  }

  int ncol() const { return ncol_; }
  int nrow() const { return nrow_; }
};

}
}

#endif // TAPAS_HOT_INSPECTOR_INTERACTION_TABLE_H__
