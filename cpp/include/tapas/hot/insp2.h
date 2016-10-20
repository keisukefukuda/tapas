#ifndef TAPAS_HOT_INSP2_H_
#define TAPAS_HOT_INSP2_H_


#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/ghost_cell.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/mapper.h>
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

  using GCell = GhostCell<Region<Dim,FP>>;

  using ProxyAttr = tapas::hot::proxy::ProxyAttr<TSP>;
  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<TSP>;

  /**
   * \brief Do inspection between a target (local) root a source (remote) root
   */
  template<class UserFunct, class...Args>
  static void DoInspect(Data &data, KeyType trg_key, KeyType src_key,
                      KeySet &req_keys_attr, KeySet &req_keys_body,
                      UserFunct f, Args...args) {
    std::cout << "*** DoInspect() : trg_key = " << trg_key << ", src_key = " << src_key << std::endl;
    const int max_level = data.max_level_;

    std::map<int, int> level_map; // target level -> source level
    // level_map is a map from target level (LS) to corresponding source levels (LS) so that
    //   if D(target cell of level LS, ghost cell of LS) is 'approximate' then
    //     
    // 
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

    // construct v_map, a map from source level to the most conservative target level
    
    //double end = MPI_Wtime();
    //data.time_rec_.Record(data.timestep_, "Map2-LET-insp", end - beg);
  }
};

}
} // namespace tapas

#endif // TAPAS_HOT_INSP2_H_
