#ifndef TAPAS_HOT_INSPECTOR_INTERACTION_H__
#define TAPAS_HOT_INSPECTOR_INTERACTION_H__

#include <vector>

#include <tapas/hot/cell.h>
#include <tapas/hot/proxy/oneside_traverse_policy.h>

namespace tapas {
namespace hot {

// Subroutines for inspector interactions
template<class TSP, class UserFunct, class ...Args>
class Interaction {
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
  
 public:
  Interaction(Data &data) : data_(data)
  {
  }

  IntrFlag TryIntr(const Reg &trg_reg, const Reg &src_reg,
                          int trg_dep, int src_dep, bool is_left_leaf,
                          UserFunct f, Args... args) {
    auto tw = data_.region_.width() / pow(2, trg_dep); // n-dimensional width of the target ghost cell
    auto sw = data_.region_.width() / pow(2, src_dep); // n-dimensional width of the source ghost cell
    
    GCell trg_gc = GCell(data_, nullptr, trg_reg, tw, trg_dep);
    GCell src_gc = GCell(data_, nullptr, src_reg, sw, src_dep);

    trg_gc.SetIsLeaf(is_left_leaf);
    src_gc.SetIsLeaf(false);

    return GCell::PredSplit2(trg_gc, src_gc, f, args...);
  }

  IntrFlag TryIntrOnSameDepth(const Reg &trg_reg, const Reg &src_reg,
                              int trg_dep, int src_dep,
                              UserFunct f, Args... args) {
    // By the nature of octrees, if the traget cell and source cell are
    // in the same depth, they have the same width (theoretically).
    // In practice, however, the floating point values are sometimes
    // slightly different.
    // The user code may use the widths to make a decision to split the source cell
    // (i.e. "split the larger one").
    // Thus, in the inspector, we have to cover all cases:
    //  (1) the two cells are exactly equal size
    //  (2) the target cell is `slightly' bigger
    //  (3) the source cell is `slightly' bigger

    // case (1)
    VecT sw = data_.region_.width() / pow(2, src_dep); // source width
    VecT tw = sw;                                      // target width

    IntrFlag sp1, sp2, sp3; // split decision result

    {
      GCell src_gc(data_, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data_, nullptr, trg_reg, tw, trg_dep);
      sp1 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }

    // case (2)
    for (int d = 0; d < Dim; d++) {
      tw[d] = sw[d] + tapas::util::NearZeroValue<FP>(sw[d]);
    }
    {
      GCell src_gc(data_, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data_, nullptr, trg_reg, tw, trg_dep);
      sp2 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }

    // case (3)
    tw = sw;
    for (int d = 0; d < Dim; d++) {
      sw[d] += tapas::util::NearZeroValue<FP>(sw[d]);
    }
    {
      GCell src_gc(data_, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data_, nullptr, trg_reg, tw, trg_dep);
      sp3 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }

    // merge the result.
    return sp1 | sp2 | sp3;
  }
};

}
}

#endif // TAPAS_HOT_INSPECTOR_INTERACTION_TABLE_H__
