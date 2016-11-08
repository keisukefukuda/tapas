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
        table[(td - trg_depth) * ncol + (sd - src_depth)] = split;
      }
    }

    return table;
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

    tapas::debug::BarrierExec([&](int,int) {
        // Repeat over all pairs target and source local trees
        for (KeyType src_key : data.gleaves_) {
          for (KeyType trg_key : data.lroots_) {
            if (src_key != trg_key) {
              const int max_depth = data.max_depth_;
              const int src_depth = SFC::GetDepth(src_key);
              const int trg_depth = SFC::GetDepth(trg_key);

              int ncol = max_depth - src_depth + 1;
              int nrow = max_depth - trg_depth + 1;
    
              std::vector<SplitType> table = BuildTable(data, src_key, trg_key, f, args...);

              TAPAS_ASSERT(table.size() == (size_t)(ncol * nrow));

#if 0
              // debug dump
              std::cout << std::endl;
              std::cout << "src = [" << src_depth << ", " << max_depth << "]" << std::endl;
              std::cout << "trg = [" << trg_depth << ", " << max_depth << "]" << std::endl;
              for (int i = 0; i < nrow; i++) {
                for (int j = 0; j < ncol; j++) {
                  SplitType split = table[i * ncol + j];
                  switch(split) {
                    case SplitType::SplitBoth:
                      std::cout << "＼";
                      break;
                    case SplitType::SplitLeft:
                      std::cout << "↓";
                      break;
                    case SplitType::SplitRight:
                      std::cout << "→";
                      break;
                    case SplitType::Approx:
                    case SplitType::Body:
                      std::cout << "・";
                      break;
                    default:
                      std::cout << "？";
                      break;
                  }
                }
                std::cout << std::endl;
              }
              std::cout << std::endl;
#endif
            }
          }
        }
      });

    // construct v_map, a map from source level to the most conservative target level
    
    //double end = MPI_Wtime();
    //data.time_rec_.Record(data.timestep_, "Map2-LET-insp", end - beg);
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSP2_H_
