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

  using ProxyCellF = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;

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
                       ? inter_.TryIntrOnSameDepth(trg_reg, src_reg, trg_depth, src_depth,
                                                          f, args...)
                       : inter_.TryIntr(trg_reg, src_reg, trg_depth, src_depth,
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

  // Traverse Global Tree
  // Since this is a source-side traversal, all source cells are in the local process.
  // (Just stop traversal if src cell is not local.)
  // Target cells are remote and approximated. However, every process has a copy of the
  // global tree and thus has target cells in the global tree.
  // TraverseGT() function traverses over
  //    [trg cells in GT] x [src cells]
  template<class Callback>
  void TraverseGT(int trg_rank, Callback callback,
                  KeyType trg_key, KeyType src_key, UserFunct f, Args...args) {
    TAPAS_ASSERT(data_.ht_.count(trg_key) > 0 || data_.ht_gtree_.count(trg_key) > 0);

    // Source cell may not be in the local process
    // Stop traversal if the src_key is not in the local process
    if (data_.ht_.count(src_key) == 0) {
      return;
    }

    TAPAS_ASSERT(data_.ht_gtree_.count(trg_key) != 0);

    IntrFlag flg = ProxyCellF::PredSplit2(trg_key, src_key, data_, f, args...);
    
    bool is_src_lf = data_.ht_[src_key]->IsLeaf();
    TAPAS_ASSERT(!(is_src_lf && flg.IsSplitR())); (void) is_src_lf;

    CellType *tc = data_.ht_.count(trg_key) > 0
                   ? data_.ht_[trg_key]
                   : data_.ht_gtree_[trg_key];
    CellType *sc = data_.ht_[src_key];
    
    TAPAS_ASSERT(tc != nullptr);
    TAPAS_ASSERT(sc != nullptr);

    bool cont = callback(trg_key, tc->IsLeaf(),
                         src_key, sc->IsLeaf(),
                         flg);
    
    // If the return value is false, stop traversing.
    if (!cont) return;

    // There are 7 patterns of recursive traversal.
    // 
    // (1) Approximate => finish traversal
    // 
    // Condition 1: trg key is a global leaf(A) or not(B)
    // Condition 2: the split flag is SplitBoth(x), SplitR(y), or SplitL(z)
    //
    // (2) A, x => TraverseApxlt (trg children, src children)
    // (3) A, y => TraverseGT    (trg,          src children)
    // (4) A, z => TraverseApxlt (trg children, src)
    // (5) B, x => TraverseGT    (trg children, src children)
    // (6) B, y => TraverseGT    (trg,          src children)
    // (7) B, z => TraverseGT    (trg children, src)

    bool is_trg_gl = data_.gleaves_.count(trg_key) > 0; // Condition 1

    // Continue traversal only if the trg_key belongs to the target rank
    bool is_tr = is_trg_gl
                 ? data_.gleaf_owners_[trg_key] == trg_rank
                 : false;

    if (flg.IsApprox()) { // (1)
      return;
    } else if (is_trg_gl && flg.IsSplitBoth()) {  // (2)
      if (!is_tr) { return; }
      ITable tbl(data_, trg_key, src_key, f, args...);
      if (!data_.ht_[src_key]->IsLeaf()) {
        for (KeyType sc : SFC::GetChildren(src_key)) {
          for (KeyType tc : SFC::GetChildren(trg_key)) {
            TraverseApxLT(tbl, callback, tc, sc, f, args...);
          }
        }
      }
    } else if (is_trg_gl && flg.IsSplitR()) {      // (3)
      for (KeyType sc : SFC::GetChildren(src_key)) {
        TraverseGT(trg_rank, callback, trg_key, sc, f, args...);
      }
    } else if (is_trg_gl && flg.IsSplitL()) {    // (4)
      if (!is_tr) { return; }
      ITable tbl(data_, trg_key, src_key, f, args...);
      for (KeyType tc : SFC::GetChildren(trg_key)) {
        TraverseApxLT(tbl, callback, tc, src_key, f, args...);
      }
    } else if (!is_trg_gl && flg.IsSplitBoth()) {  // (5) 
      for (KeyType tc : SFC::GetChildren(trg_key)) {
        for (KeyType sc : SFC::GetChildren(src_key)) {
          TraverseGT(trg_rank, callback, tc, sc, f, args...);
        }
      }
    } else if (!is_trg_gl && flg.IsSplitR()) {     // (6)
      for (KeyType sc : SFC::GetChildren(src_key)) {
        TraverseGT(trg_rank, callback, trg_key, sc, f, args...);
      }
    } else if (!is_trg_gl && flg.IsSplitL()) {     // (7)
      for (KeyType tc : SFC::GetChildren(trg_key)) {
        TraverseGT(trg_rank, callback, tc, src_key, f, args...);
      }
    } else {
      // this should not happen
      TAPAS_ASSERT(0);
    }
  }

  template<class Callback>
  void TraverseApxLT(ITable &tbl, Callback callback,
                     KeyType trg_root_key, KeyType src_key,
                     UserFunct f, Args...args) {
    if (data_.ht_.count(src_key) == 0) {
      return;
    }

    if (data_.ht_[src_key]->IsLeaf()) {
      return;
    }
    
    int trg_root_depth = SFC::GetDepth(trg_root_key);
    int src_depth = SFC::GetDepth(src_key);
    int nrows = data_.max_depth_ - trg_root_depth + 1;

    IntrFlag flg; // split type flag

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

        IntrFlag sp2 = inter_.RetryIntr(trg_reg, src_reg, trg_depth, src_depth, sp, f, args...);

        flg.Add(sp2);
      } else {
        flg.Add(sp);
      }
    }

    bool is_src_leaf = data_.ht_[src_key]->IsLeaf();
    bool cont = callback(trg_root_key, false, src_key, is_src_leaf, flg);

    if (cont && !is_src_leaf) {
      for (KeyType sc : SFC::GetChildren(src_key)) {
        TraverseApxLT(tbl, callback, trg_root_key, sc, f, args...);
      }
    }
  }
    
  /**
   * \brief Inspector for Map-2. Traverse the local source trees
   * using pseudo(approximated) target cells
   *
   * \param rank of the process of which cells are traversed
   */
  template<class Callback>
  void Inspect(int rank, KeyType root, Callback &callback, UserFunct f, Args...args) {
    // First, traverse global tree and source cells.
    // The global tree is shared by all processes, so all processes need
    // the traversed source cells.

    TraverseGT(rank, callback, root, root, f, args...);
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSPECTOR_ONESIDE_ON_SOURCE_H__
