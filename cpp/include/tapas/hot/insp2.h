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
   * \brief Do inspection between a target (local) root a source (remote) root
   */
  template<class UserFunct, class...Args>
  static void DoInspect(Data &data, KeyType trg_key, KeyType src_key,
                        KeySet &/*req_keys_attr*/, KeySet &/*req_keys_body*/,
                        UserFunct /*f*/, Args.../*args*/) {
    const int max_depth = data.max_depth_;

    int src_depth = SFC::GetDepth(src_key);
    int trg_depth = SFC::GetDepth(trg_key);

    // std::cout << "*** DoInspect() : trg_key = " << trg_key << "(" << trg_depth << "), "
    //           << "src_key = " << src_key << "(" << src_depth << "), "
    //           << "max_depth = " << max_depth
    //           << std::endl;

    Reg src_reg = SFC::CalcRegion(src_key, data.region_);
    Reg trg_reg = SFC::CalcRegion(trg_key, data.region_);

    for (int sd = src_depth; sd <= max_depth; sd++) {
      for (int td = trg_depth; td <= max_depth; td++) {
        auto sw = data.region_.width(); // n-dim dimension width of the source ghost cell
        auto tw = data.region_.width(); // n-dim dimension width of the target ghost cell
        for (int d = 0; d < sd; d++) { sw /= 2; }
        for (int d = 0; d < td; d++) { tw /= 2; }

        //std::cout << sd << "-" << td << " ";

        GCell src_gc = GCell(data, nullptr, src_reg, sw, sd);
        GCell trg_gc = GCell(data, nullptr, src_reg, tw, td);
        
        //f(trg_gc, src_gc, args...);

      }
    }
    //std::cout << std::endl;
    
    std::unordered_map<int, int> level_map; // Source level -> target level
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

    // std::cout << tapas::mpi::Rank() << " Insp2::Inspect" << std::endl;
    
    for (KeyType src_key : data.gleaves_) {
      for (KeyType trg_key : data.lroots_) {
        if (src_key != trg_key) {
          DoInspect(data, trg_key, src_key, req_keys_attr, req_keys_body, f, args...);
        }
      }
    }

    // construct v_map, a map from source level to the most conservative target level
    
    //double end = MPI_Wtime();
    //data.time_rec_.Record(data.timestep_, "Map2-LET-insp", end - beg);
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSP2_H_
