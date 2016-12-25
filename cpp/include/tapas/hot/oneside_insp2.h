
#ifndef TAPAS_HOT_ONESIDE_INSP2_H_
#define TAPAS_HOT_ONESIDE_INSP2_H_

/*
  Implementation memo:
  SrcSideInspectorと，リスト作成のロジックを分離する
  まず，アクションを関数として分離する
  アクション = リストを作成する
  
  作業順
  [ ] InteractionTypeを拡張する．attrの読み取り，bodyの読み取りについても変数を定義する
  [ ] 複数に分かれている分割種類の定数を，InteractionTypeに統合
  [ ] Inspector用のルーチンが，ReadAttr, ReadBodies についても正しい結果を返すように変更する
  [ ] Actionクラスを定義し，class InspActionLET を定義する．req_attr, req_body をメンバー変数として持つクラスとして定義する
*/

#include <unordered_map>

#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/proxy/oneside_traverse_policy.h>
#include <tapas/hot/proxy/proxy_cell.h>
#include <tapas/iterator.h>

namespace tapas {
namespace hot {

/**
 * \brief Inspector implementation for Map-2
 */
template<class TSP>
class OnesideInsp2 {
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

  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<ProxyCell>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<ProxyCell>;

  /**
   * Inspecting action for LET construction
   */
  class InspAction {
   public:
    inline void operator()(KeyType trg, KeyType src, InteractionType s) {
      (void)trg; (void)src; (void)s;
    }
  };
  
  template<class UserFunct, class...Args>
  static InteractionType TryInteraction(Data &data,
                                  const Reg &trg_reg, const Reg &src_reg,
                                  int trg_dep, int src_dep, bool is_left_leaf,
                                  UserFunct f, Args... args) {
    auto sw = data.region_.width() / pow(2, src_dep); // n-dimensional width of the source ghost cell
    auto tw = data.region_.width() / pow(2, trg_dep); // n-dimensional width of the target ghost cell
        
    GCell src_gc = GCell(data, nullptr, src_reg, sw, src_dep);
    GCell trg_gc = GCell(data, nullptr, trg_reg, tw, trg_dep);

    trg_gc.SetIsLeaf(is_left_leaf);
    
    return GCell::PredSplit2(trg_gc, src_gc, f, args...);
  }
  
  template<class UserFunct, class...Args>
  static InteractionType TryInteractionOnSameLevel(Data &data,
                                             const Reg &trg_reg, const Reg &src_reg,
                                             int trg_dep, int src_dep,
                                             UserFunct f, Args... args) {
    // By the nature of octrees, if the traget and source depth are equal,
    // target and source cells have the same width theoretically.
    // In practice, however, the floating point values are sometimes
    // slightly different.
    // The user code may use the values to make a decision to split the source cell
    // (i.e. "split the larger one").
    // Thus, in the inspector, we have to cover all cases:
    //  (1) the two cells are of exactly equal size
    //  (2) the target cell is `slightly' bigger
    //  (3) the source cell is `slightly' bigger
    
    // case 1.
    VecT sw = data.region_.width() / pow(2, src_dep);
    VecT tw = sw;

    InteractionType split1, split2, split3; // split decision result

    {
      GCell src_gc(data, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data, nullptr, trg_reg, tw, trg_dep);
      split1 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }
    
    // case 2.
    for (int d = 0; d < Dim; d++) {
      tw[d] = sw[d] + tapas::util::NearZeroValue<FP>(sw[d]);
    }
    {
      GCell src_gc(data, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data, nullptr, trg_reg, tw, trg_dep);
      split2 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }

    // case 3.
    tw = sw;
    for (int d = 0; d < Dim; d++) {
      sw[d] += tapas::util::NearZeroValue<FP>(sw[d]);
    }
    {
      GCell src_gc(data, nullptr, src_reg, sw, src_dep);
      GCell trg_gc(data, nullptr, trg_reg, tw, trg_dep);
      split3 = GCell::PredSplit2(trg_gc, src_gc, f, args...);
    }

    // merge the result.
    int sp = static_cast<int>(split1)
             | static_cast<int>(split2)
             | static_cast<int>(split3);

    // Note: We don't think about leaf-case in this function
    //       because if one or both of the trg & src cell is/are leaf,
    //       the 'slight difference' issue doesn't occur.
    if      (sp & static_cast<int>(InteractionType::SplitBoth))  { return InteractionType::SplitBoth; }
    else if (sp & static_cast<int>(InteractionType::SplitLeft))  { return InteractionType::SplitLeft; }
    else if (sp & static_cast<int>(InteractionType::SplitRight)) { return InteractionType::SplitRight; }

    return InteractionType::Approx;
  }

  /**
   * \brief Do inspection between a single pair of target (local) root and source (remote) root
   *
   * \param data Data
   * \param src_key Key of a local rootof the source (remote) process
   * \param trg_key Key of a local root of the target (local) process
   * \param f The user function
   * \param args Arguments to the user function
   */
  template<class UserFunct, class...Args>
  static std::vector<InteractionType> BuildTable(Data &data, KeyType src_root_key, KeyType trg_root_key, UserFunct f, Args...args) {
    const int max_depth = data.max_depth_;
    const int src_depth = SFC::GetDepth(src_root_key);
    const int trg_depth = SFC::GetDepth(trg_root_key);
    
    const Reg src_reg = SFC::CalcRegion(src_root_key, data.region_);
    const Reg trg_reg = SFC::CalcRegion(trg_root_key, data.region_);

    const int ncol = max_depth - src_depth + 1;
    const int nrow = max_depth - trg_depth + 1;
    std::vector<InteractionType> table(ncol * nrow);

    for (int sd = src_depth; sd <= max_depth; sd++) { // source depth
      for (int td = trg_depth; td <= max_depth; td++) { // target depth
        InteractionType split;
        
        if (td == sd) {
          split = TryInteractionOnSameLevel(data, trg_reg, src_reg, td, sd, f, args...);
        } else {
          split = TryInteraction(data, trg_reg, src_reg, td, sd, false, f, args...);
        }

        // We here use a ghost cell for the target cell and assume that
        // the target ghost cell is not a leaf.
        // Thus, there is possibility that the source cell is split
        // even if the `split` value above is not SplitRight if the target cell is a leaf.
        if (split != InteractionType::SplitRight
            && split != InteractionType::SplitBoth
            && td >= data.min_leaf_level_[trg_root_key]) {
          InteractionType split2 = TryInteraction(data, trg_reg, src_reg, td, sd, true, f, args...);
          
          TAPAS_ASSERT(split2 != InteractionType::SplitBoth); // because left cell (target) is a leaf.
          TAPAS_ASSERT(split2 != InteractionType::SplitLeft);

          if (split2 == InteractionType::SplitRight) {
            split = InteractionType::SplitRightILL;
          }
        }
        int idx = (td - trg_depth) * ncol + (sd - src_depth);
        table[idx] = split;
      }
    }

    return table;
  }

  static void DumpTable(const std::vector<InteractionType> &table, int nrow, int ncol) {
    // debug dump
    std::cout << " ";
    for (int i = 0; i < ncol; i++) {
      std::cout << i;
    }
    std::cout << " S" << std::endl;
    
    for (int i = 0; i < nrow; i++) {
      std::cout << i;
      for (int j = 0; j < ncol; j++) {
        InteractionType split = table[i * ncol + j];
        switch(split) {
          case InteractionType::SplitBoth:
            std::cout << "\\";
            break;
          case InteractionType::SplitLeft:
            std::cout << "|";
            break;
          case InteractionType::SplitRight:
            std::cout << "-";
            break;
          case InteractionType::SplitRightILL:
            std::cout << "+";
            break;
          case InteractionType::Approx:
          case InteractionType::Body:
            std::cout << "*";
            break;
          default:
            std::cout << "?";
            break;
        }
      }
      std::cout << std::endl;
    }
    std::cout << "T" << std::endl;
  }

  /**
   * \brief Traverse the soruce tree and collect 
   */
  template<class UserFunct, class...Args>
  static void TraverseSource(Data &data, std::vector<InteractionType> &table,
                             KeySet &req_keys_attr, KeySet &req_keys_body,
                             KeyType trg_root_key, // key of the root of the target local tree
                             KeyType src_root_key, // key of the root of the source local tree
                             KeyType src_key,
                             UserFunct f, Args...args) {
    if (data.shallow_leaves_.count(src_key) > 0) {
      req_keys_body.insert(src_key);
      return;
    }
    
    int src_root_depth = SFC::GetDepth(src_root_key);
    int trg_root_depth = SFC::GetDepth(trg_root_key);
    int src_depth = SFC::GetDepth(src_key);
    int ncols = data.max_depth_ - src_root_depth + 1;
    int nrows = data.max_depth_ - trg_root_depth + 1;

    int c = src_depth - src_root_depth;

    int cnt = 0;
    for (int r = 0; r < nrows; r++) {
      auto sp = table[r * ncols + c];

      if (sp == InteractionType::SplitRight
          || sp == InteractionType::SplitBoth
          || sp == InteractionType::SplitRightILL) {
        // We found the source cell (closest ghost source cell) is split.
        // We need to check if the real cell (src_key) is split.
        // The real cell is farther from the target cell than the ghost cell,
        // So  Far(T, Ghost cell) => Far
        //     Near(T, Ghost cell) => Near or Far <- it's the case here.
        // If they are Far, we don't need to split the cell and transfer the children of the source cell.
        
        int trg_depth = trg_root_depth + r;
        const Reg trg_reg = SFC::CalcRegion(trg_root_key, data.region_);
        const Reg src_reg = SFC::CalcRegion(src_key, data.region_);

        InteractionType sp2 = (src_depth == trg_depth)
                        ? TryInteractionOnSameLevel(data, trg_reg, src_reg, trg_depth, src_depth, f, args...)
                        : TryInteraction(data, trg_reg, src_reg, trg_depth, src_depth, sp == InteractionType::SplitRightILL, f, args...);
        
        if (sp2 == InteractionType::SplitRight || sp2 == InteractionType::SplitBoth) {
          // source side is still split.
          cnt++;
        }
      }
    }
 
    req_keys_attr.insert(src_key);
    
    if (cnt == 0 && data.ht_.count(src_key) == 0) {
      if (src_depth == data.max_depth_) {
        req_keys_body.insert(src_key);
      } else {
        req_keys_attr.insert(src_key);
      }
    } else {
      for (auto ch_key : SFC::GetChildren(src_key)) {
        TraverseSource(data, table, req_keys_attr, req_keys_body, trg_root_key, src_root_key, ch_key, f, args...);
      }
    }

    return;
  }

  /**
   * \brief Inspector for Map-2. Traverse hypothetical global tree and 
   *        construct a cell list to be exchanged between processes.
   */
  template<class UserFunct, class...Args>
  static void Inspect(CellType &root,
                      KeySet &req_keys_attr, KeySet &req_keys_body,
                      UserFunct f, Args...args) {

    auto &data = root.data();
    req_keys_attr.clear(); // cells of which attributes are to be transfered from remotes to local
    req_keys_body.clear(); // cells of which bodies are to be transfered from remotes to local

    std::mutex list_attr_mutex, list_body_mutex;

    // Construct request lists of necessary cells
    req_keys_attr.insert(root.key());

    // Repeat over all pairs target and source local trees
    for (KeyType src_key : data.gleaves_) {
      if (data.ht_.count(src_key) == 0) {
        for (KeyType trg_key : data.lroots_) {
          if (src_key != trg_key) {
            //double bt = MPI_Wtime();
            const int max_depth = data.max_depth_;
            const int src_depth = SFC::GetDepth(src_key);
            const int trg_depth = SFC::GetDepth(trg_key);

            int ncol = max_depth - src_depth + 1;
            int nrow = max_depth - trg_depth + 1;

            std::vector<InteractionType> table = BuildTable(data, src_key, trg_key, f, args...);

            TAPAS_ASSERT(table.size() == (size_t)(ncol * nrow)); (void)nrow; (void)ncol;

            // If table[0] is "approximate", We don't need to traverse the source local tree.
            // (NOTE: all local roots are leaves of the global tree, thus all local roots are shared
            // among all processes, no need to transfer)
            if (table[0] == InteractionType::Approx) {
              continue;
            }

            //double et = MPI_Wtime();
            //std::cout << "Inner-most loop took " << std::scientific << (et-bt) << std::endl;
            TraverseSource(data, table, req_keys_attr, req_keys_body, trg_key, src_key, src_key, f, args...);
          }
        }
      }
    }
    // construct v_map, a map from source level to the most conservative target level
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_ONESIDE_INSP2_H_
