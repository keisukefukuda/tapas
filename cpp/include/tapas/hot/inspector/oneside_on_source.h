#ifndef TAPAS_HOT_INSPECTOR_ONESIDE_ON_SOURCE_H__
#define TAPAS_HOT_INSPECTOR_ONESIDE_ON_SOURCE_H__

/*
  Implementation memo:
  SrcSideInspectorと，リスト作成のロジックを分離する
  まず，アクションを関数として分離する
  アクション = リストを作成する

  作業順
  [x] InteractionTypeを拡張する．attrの読み取り，bodyの読み取りについても変数を定義する
  [x] 複数に分かれている分割種類の定数を，InteractionTypeに統合
  [x] Inspector用のルーチンが，ReadAttr, ReadBodies についても正しい結果を返すように変更する
  [ ] Actionクラスを定義し，class InspActionLET を定義する．req_attr, req_body をメンバー変数として持つクラスとして定義する
*/

#include <unordered_map>

#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/inspector/interaction.h>
#include <tapas/hot/inspector/interaction_table.h>
#include <tapas/hot/inspector/interaction.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/proxy/oneside_traverse_policy.h>
#include <tapas/hot/proxy/proxy_body.h>
#include <tapas/hot/proxy/proxy_cell.h>
#include <tapas/iterator.h>

namespace tapas {
namespace hot {

/**
 * \brief Inspector implementation for Map-2
 * 
 * One-sided inspector on target processes
 */
template<class TSP, class UserFunct, class... Args>
class OnesideOnSource {
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

  using ITable = InteractionTable<TSP, UserFunct, Args...>;

 private:
  Data &data_;
  Interaction<TSP, UserFunct, Args...> inter_;
  
 public:
  OnesideOnSource(Data &data)
      : data_(data)
      , inter_(data)
  {
  }

#if 0
  static void DumpTable(const std::vector<IntrFlag> &table, int nrow, int ncol) {
    // debug dump
    std::cout << " ";
    for (int i = 0; i < ncol; i++) {
      std::cout << i;
    }
    std::cout << " S" << std::endl;

    for (int i = 0; i < nrow; i++) {
      std::cout << i;
      for (int j = 0; j < ncol; j++) {
        IntrFlag split = table[i * ncol + j];
        if (split.IsSplitBoth()) {
          // split both
          std::cout << "\\";
        } else if (split.IsSplitL()) {
          // split left
          std::cout << "|";
        } else if (split.IsSplitR) {
          // split right
          std::cout << "-";
        } else {
          // seems to be approx.
          std::cout << "*";
        }
      }
      std::cout << std::endl;
    }
    std::cout << "T" << std::endl;
  }
#endif

  /**
   * \brief Traverse the soruce tree and collect
   */
  template<class Callback>
  void TraverseSource(ITable &tbl,
                      Callback &callback,
                      KeyType trg_root_key, // key of the root of the target local tree
                      KeyType src_root_key, // key of the root of the source local tree
                      KeyType src_key,
                      UserFunct f, Args...args) {
    int trg_root_depth = SFC::GetDepth(trg_root_key);
    int src_depth = SFC::GetDepth(src_key);
    int nrows = data_.max_depth_ - trg_root_depth + 1;

    IntrFlag flag; // split type flag

    for (int r = 0; r < nrows; r++) { // rows are target depth
      int trg_depth = trg_root_depth + r;
      auto sp = tbl.At(trg_depth, src_depth);

      if (sp.IsSplitR() || sp.IsSplitILL()) {
        // We found the source cell (closest ghost source cell) is split.
        // We need to check if the real cell (src_key) is split.
        // The real cell is farther from the target cell than the ghost cell,
        // So  Far(T, Ghost cell) => Far
        //     Near(T, Ghost cell) => Near or Far <- it's the case here.
        // If they are Far, we don't need to split the cell and transfer the children of the source cell.

        const Reg trg_reg = SFC::CalcRegion(trg_root_key, data_.region_);
        const Reg src_reg = SFC::CalcRegion(src_key, data_.region_);

        IntrFlag sp2 = (src_depth == trg_depth)
                       ? inter_.TryInteractionOnSameLevel(trg_reg, src_reg, trg_depth, src_depth,
                                                          f, args...)
                       : inter_.TryInteraction(trg_reg, src_reg, trg_depth, src_depth,
                                               sp.IsSplitILL(), f, args...);
        flag.Add(sp2);
      } else {
        flag.Add(sp);
      }
    }

    bool is_src_leaf = (src_depth == data_.max_depth_);
    bool cont = callback(trg_root_key, false, src_key, is_src_leaf, flag);

    if (cont && SFC::GetDepth(src_key) < data_.max_depth_) {
      for (auto ch_key : SFC::GetChildren(src_key)) {
        TraverseSource(tbl, callback, trg_root_key, src_root_key, ch_key, f, args...);
      }
    }
    return;
  }

  /**
   * \brief Inspector for Map-2. Traverse the local source trees
   * using pseudo(approximated) target cells
   */
  template<class Callback>
  void Inspect(CellType &root, Callback &callback,
               UserFunct f, Args...args) {
    auto &data = root.data();

    // Start source-side traverse from
    //   traget key : root
    //   source key : global leaves
    KeyType trg_key = 0; // root

    for (KeyType src_key : data.gleaves_) {
      if (data.ht_.count(src_key) == 0) {
        ITable tbl(data_, trg_key, src_key, f, args...);

        if (tbl.IsRootApprox()) {
          continue;
        }

        TraverseSource(tbl, callback, trg_key, src_key, src_key, f, args...);
      }
    }
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSPECTOR_ONESIDE_ON_SOURCE_H__
