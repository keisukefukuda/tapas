
#ifndef TAPAS_HOT_INSP2_H_
#define TAPAS_HOT_INSP2_H_

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
class Insp2 {
 public:
  
  // typedefs
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
  using Vec = tapas::Vec<TSP::Dim, typename TSP::FP>;
  using Reg = Region<Dim, FP>;

  using TravPolicy = tapas::hot::proxy::OnesideTraversePolicy<Dim, FP, Data>;
  using GCell = tapas::hot::proxy::ProxyCell<TSP, TravPolicy>;

  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<ProxyCell>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<ProxyCell>;

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
  static std::vector<SplitType> BuildTable(Data &data, KeyType src_key, KeyType trg_key, UserFunct f, Args...args) {
    const int max_depth = data.max_depth_;
    const int src_depth = SFC::GetDepth(src_key);
    const int trg_depth = SFC::GetDepth(trg_key);
    
    const Reg src_reg = SFC::CalcRegion(src_key, data.region_);
    const Reg trg_reg = SFC::CalcRegion(trg_key, data.region_);

    const int ncol = max_depth - src_depth + 1;
    const int nrow = max_depth - trg_depth + 1;
    std::vector<SplitType> table(ncol * nrow);
    
    for (int sd = src_depth; sd <= max_depth; sd++) {
      for (int td = trg_depth; td <= max_depth; td++) {
        auto sw = data.region_.width() / pow(2, sd); // n-dim dimensional width of the source ghost cell
        auto tw = data.region_.width() / pow(2, td); // n-dim dimensional width of the target ghost cell

        GCell src_gc = GCell(data, nullptr, src_reg, sw, sd);
        GCell trg_gc = GCell(data, nullptr, trg_reg, tw, td);

        SplitType split = GCell::PredSplit2(trg_gc, src_gc, f, args...);
        int idx = (td - trg_depth) * ncol + (sd - src_depth);
        table[idx] = split;

        // We here use a ghost cell for the target cell and assume that
        // the target ghost cell is not a leaf.
        // Thus, there is possibility that the source cell is split
        // even if the `split` value above is not SplitRight if the target cell is a leaf.
        if (table[idx] != SplitType::SplitRight && table[idx] != SplitType::SplitBoth
            && td >= data.min_leaf_level_[trg_key]) {
          trg_gc.ClearFlags();
          trg_gc.SetIsLeaf(true); // re-use the target Ghost Cell, but now a leaf

          if (trg_key == 1152921504606846977 && src_key == 3458764513820540929) {
            setenv("TAPAS_DEBUG_INSPECTOR", "1", 1);
            std::cout << std::endl;
          }
          SplitType split = GCell::PredSplit2(trg_gc, src_gc, f, args...);
          if (trg_key == 1152921504606846977 && src_key == 3458764513820540929) {
            unsetenv("TAPAS_DEBUG_INSPECTOR");
            std::cout << "Re-checking " << td << ", " << sd << " : " << table[idx] << " -> " << split << std::endl;
          }

          TAPAS_ASSERT(split != SplitType::SplitBoth); // because left cell (target) is a leaf.
          TAPAS_ASSERT(split != SplitType::SplitLeft);

          if (split == SplitType::SplitRight) {
            table[idx] = SplitType::SplitRightILL;
          }
        }
      }
    }

    return table;
  }

  static void DumpTable(const std::vector<SplitType> &table, int nrow, int ncol) {
    // debug dump
    std::cout << " ";
    for (int i = 0; i < ncol; i++) {
      std::cout << i;
    }
    std::cout << std::endl;
    
    for (int i = 0; i < nrow; i++) {
      std::cout << i;
      for (int j = 0; j < ncol; j++) {
        SplitType split = table[i * ncol + j];
        switch(split) {
          case SplitType::SplitBoth:
            std::cout << "\\";
            break;
          case SplitType::SplitLeft:
            std::cout << "|";
            break;
          case SplitType::SplitRight:
            std::cout << "-";
            break;
          case SplitType::SplitRightILL:
            std::cout << "+";
            break;
          case SplitType::Approx:
          case SplitType::Body:
            std::cout << "*";
            break;
          default:
            std::cout << "?";
            break;
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  /**
   * \brief Traverse the soruce tree and collect 
   */
  template<class UserFunct, class...Args>
  static void TraverseSource(Data &data, std::vector<SplitType> &table,
                             KeySet &req_keys_attr, KeySet &req_keys_body,
                             KeyType trg_root_key, // key of the root of the target local tree
                             KeyType src_root_key, // key of the root of the source local tree
                             KeyType src_key,
                             UserFunct f, Args...args) {
    int src_root_depth = SFC::GetDepth(src_root_key);
    int trg_root_depth = SFC::GetDepth(trg_root_key);
    int src_depth = SFC::GetDepth(src_key);
    int ncols = data.max_depth_ - src_root_depth + 1;
    int nrows = data.max_depth_ - trg_root_depth + 1;

    bool debug = (trg_root_key == SFC::Parent(SFC::Parent(1630303065108119555)))
                 && (src_key == 3650167497983787012);
    if (debug) {
      std::cout << "In TraverseSource trg_root_key=" << SFC::Decode(trg_root_key) << " = " << trg_root_key << std::endl;
      std::cout << "In TraverseSource src_root_key=" << SFC::Decode(src_root_key) << " = " << src_root_key << std::endl;
      std::cout << "In TraverseSource src_key     =" << SFC::Decode(src_key)  << " = " << src_key << std::endl;
    }
    
    // std::cout << "----------------------------------" << std::endl;
    // std::cout << "trg root = " << SFC::Decode(trg_root_key) << std::endl;
    // std::cout << "src root = " << SFC::Decode(src_root_key) << std::endl;
    // std::cout << "src key  = " << SFC::Decode(src_key) << std::endl;
    // std::cout << "src depth = " << src_depth << std::endl;
    // std::cout << "nrows = " << nrows << ", ncols = " << ncols << std::endl;
    
    //DumpTable(table, nrows, ncols);

    int c = src_depth - src_root_depth;

    //std::cout << "Checking col " << c << std::endl;
    int cnt = 0;
    for (int r = 0; r < nrows; r++) {
      auto sp = table[r * ncols + c];
      if (debug) {
        std::cout << "Table[r=" << r << ", c=" << c << "] = " << table[r * ncols + c] << std::endl;
      }
      
      if (sp == SplitType::SplitRight || sp == SplitType::SplitBoth) {
        // We found the source cell (closest ghost source cell) is split.
        // We need to check if the real cell (src_key) is split.
        // The real cell is farther from the target cell than the ghost cell,
        // So  Far(T, Ghost cell) => Far
        //     Near(T, Ghost cell) => Near or Far <- it's the case here.
        // If they are Far, we don't need to split the cell and transfer the children of the source cell.
        
        int trg_depth = trg_root_depth + r;
        const auto trg_width = data.region_.width() / pow(2, trg_depth);
        const Reg trg_reg = SFC::CalcRegion(trg_root_key, data.region_);
        const Reg src_reg = SFC::CalcRegion(src_key, data.region_);
        
        GCell trg_gc = GCell(data, nullptr, trg_reg, trg_width, trg_depth);
        GCell src_gc = GCell(data, nullptr, src_reg, src_reg.width(), src_depth);

        SplitType split = GCell::PredSplit2(trg_gc, src_gc, f, args...);

        //std::cout << ((split == SplitType::SplitRight || split == SplitType::SplitBoth) ? "Split" : "NOT Split!!");
        //std::cout << std::endl;

        if (debug) {
          std::cout << "Real check: trg_depth=" << trg_depth << ", src_depth=" << src_depth << ", result=" << split << std::endl;
          std::cout << "Ci.R = " << trg_gc.width() << std::endl;
          std::cout << "Cj.R = " << src_gc.width() << std::endl;
          std::cout << "Distance2 : " << trg_gc.Distance(src_gc, tapas::Center) << std::endl;
        }
        
        if (split == SplitType::SplitRight || split == SplitType::SplitBoth) {
          // source side is still split.
          cnt++;
        }
      }
    }
    //std::cout << "There are(is) " << cnt << " split." << std::endl;
    if (debug) {
      std::cout << "So, cnt=" << cnt << std::endl;
    }

    req_keys_attr.insert(src_key);
    if (cnt == 0 && data.ht_.count(src_key) == 0) {
      if (src_depth == data.max_depth_) {
        //if (debug) std::cout << "Body" << std::endl;
        req_keys_body.insert(src_key);
      } else {
        //if(debug) std::cout << "Attr" << std::endl;
        req_keys_attr.insert(src_key);
      }
    } else {
      for (auto ch_key : SFC::GetChildren(src_key)) {
        //if(debug) std::cout << "Recursive TraverseSource(ch_key=" << ch_key << ")" << std::endl;
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
        if (src_key == 3458764513820540929) {
          std::cout << "==========" << std::endl;
          std::cout << "In rank " << tapas::mpi::Rank() << " : Looking at " << 3458764513820540929 << "'s subtree." << std::endl;
        }
        for (KeyType trg_key : data.lroots_) {
          if (src_key != trg_key) {
            const int max_depth = data.max_depth_;
            const int src_depth = SFC::GetDepth(src_key);
            const int trg_depth = SFC::GetDepth(trg_key);

            int ncol = max_depth - src_depth + 1;
            int nrow = max_depth - trg_depth + 1;
    
            std::vector<SplitType> table = BuildTable(data, src_key, trg_key, f, args...);

#if 1
            if (src_key == 3458764513820540929 && trg_key == SFC::Parent(SFC::Parent(1630303065108119555))) {
              DumpTable(table, nrow, ncol);
            }
#endif
            
            TAPAS_ASSERT(table.size() == (size_t)(ncol * nrow));

            // If table[0] is "approximate", We don't need to traverse the source local tree.
            // (NOTE: all local roots are leaves of the global tree, thus all local roots are shared
            // among all processes, no need to transfer)
            if (table[0] == SplitType::Approx) {
              continue;
            }

            TraverseSource(data, table, req_keys_attr, req_keys_body, trg_key, src_key, src_key, f, args...);
          }
        }
        if (src_key == 3458764513820540929) {
          std::cout << "==========" << std::endl;
        }
      }
    }
    // construct v_map, a map from source level to the most conservative target level
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSP2_H_
