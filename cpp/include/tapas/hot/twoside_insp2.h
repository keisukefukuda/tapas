#ifndef TAPAS_HOT_TWOSIDE_INSP2_H_
#define TAPAS_HOT_TWOSIDE_INSP2_H_

namespace tapas {
namespace hot {

/**
 * \brief Two-side LET inspector
 */
template<class TSP>
class TwosideInsp2 {
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

  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<ProxyCell>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<ProxyCell>;
  
  /**
   * \brief Inspector for Map-2. Traverse hypothetical global tree and construct a cell list.
   */
  template<class UserFunct, class...Args>
  static void Inspect(CellType &root,
                      KeySet &req_keys_attr, KeySet &req_keys_body,
                      UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-Traverse", SCOREP_USER_REGION_TYPE_FUNCTION);
    MPI_Barrier(MPI_COMM_WORLD);

    double beg = MPI_Wtime();
    auto & data = root.data();
    req_keys_attr.clear(); // cells of which attributes are to be transfered from remotes to local
    req_keys_body.clear(); // cells of which bodies are to be transfered from remotes to local

    std::mutex list_attr_mutex, list_body_mutex;

    // Construct request lists of necessary cells
    req_keys_attr.insert(root.key());

    Traverse(root.key(), root.key(), root.data(), req_keys_attr, req_keys_body, list_attr_mutex, list_body_mutex, f, args...);

    double end = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-insp", end - beg);
  }

  /**
   * \brief Traverse a virtual global tree and collect cells to be requested to other processes.
   * \param p Traget particle
   * \param key Source cell key
   * \param data Data
   * \param list_attr (output) Set of request keys of which attrs are to be sent
   * \param list_body (output) Set of request keys of which bodies are to be sent
   */
  template<class UserFunct, class...Args>
  static void Traverse(KeyType trg_key, KeyType src_key, Data &data,
                       KeySet &list_attr, KeySet &list_body,
                       std::mutex &list_attr_mutex, std::mutex &list_body_mutex,
                       UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-Traverse", SCOREP_USER_REGION_TYPE_FUNCTION);

    using Th = typename CellType::Threading;

    // Traverse traverses the hypothetical global tree and constructs a list of
    // necessary cells required by the local process.
    auto &ht = data.ht_; // hash table

    // (A) check if the trg cell is local (kept in this function)
    if (ht.count(trg_key) == 0) {
      return;
    }

    // Maximum depth of the tree.
    const int max_depth = data.max_depth_;

    bool is_src_local = ht.count(src_key) != 0; // CAUTION: even if is_src_local, the children are not necessarily all local.
    bool is_src_local_leaf = is_src_local && ht[src_key]->IsLeaf();
    bool is_src_remote_leaf = !is_src_local && SFC::GetDepth(src_key) >= max_depth;

    //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << std::endl;

    if (is_src_local_leaf) {
      // the cell is local. everythig's fine. nothing to do.
      //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_local_leaf" << std::endl;
      return;
    }
    
    if (is_src_remote_leaf) {
      // If the source cell is a remote leaf, we need it (with it's bodies).
      list_attr_mutex.lock();
      list_attr.insert(src_key);
      list_attr_mutex.unlock();

      list_body_mutex.lock();
      list_body.insert(src_key);
      list_body_mutex.unlock();
      //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_remote_leaf" << std::endl;
      return;
    }
    TAPAS_ASSERT(SFC::GetDepth(src_key) <= SFC::MAX_DEPTH);
    list_attr_mutex.lock();
    list_attr.insert(src_key);
    list_attr_mutex.unlock();

    // Approx/Split branch
    IntrFlag split = ProxyCell::PredSplit2(trg_key, src_key, data, f, args...); // automated predicator object

    std::cout << "src_key=" << src_key << " trg_key=" << trg_key << " " << split.ToString() << std::endl;

    const constexpr int kNspawn = 3;
    bool to_spawn = SFC::GetDepth(trg_key) < kNspawn && SFC::GetDepth(src_key) < kNspawn;
    to_spawn = false;

    if (split.IsSplit()) {
      if (to_spawn) {
        typename Th::TaskGroup tg;
        for (KeyType trg_ch : SFC::GetChildren(trg_key)) {
          if (ht.count(trg_ch) > 0) {
            for (KeyType src_ch : SFC::GetChildren(src_key)) {
              tg.createTask([&]() mutable {
                  Traverse(trg_ch, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
                });
            }
          }
        }
        tg.wait();
      } else {
        for (KeyType trg_ch : SFC::GetChildren(trg_key)) {
          if (ht.count(trg_ch) > 0) {
            for (KeyType src_ch : SFC::GetChildren(src_key)) {
              Traverse(trg_ch, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
            }
          }
        }
      }
    } else if (split.IsSplitL()) {
      if (to_spawn) {
        typename Th::TaskGroup tg;
        for (KeyType ch : SFC::GetChildren(trg_key)) {
          if (ht.count(ch) > 0) {
            tg.createTask([&]() mutable {
                Traverse(ch, src_key, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
              });
          }
        }
        tg.wait();
      } else {
        for (KeyType ch : SFC::GetChildren(trg_key)) {
          if (ht.count(ch) > 0) {
            Traverse(ch, src_key, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
          }
        }
      }
    } else if (split.IsSplitR()) {
      if (to_spawn) {
        typename Th::TaskGroup tg;
        for (KeyType src_ch : SFC::GetChildren(src_key)) {
          tg.createTask([&]() mutable {
              Traverse(trg_key, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
            });
        }
        tg.wait();
      } else {
        for (KeyType src_ch : SFC::GetChildren(src_key)) {
          Traverse(trg_key, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
        }
      }
    } else {
      // Approximate
      assert(split.IsApprox());
      list_attr_mutex.lock();
      list_attr.insert(src_key);
      list_attr_mutex.unlock();
    }
    return;
  }
};

} // namespace hot
} // namespace tapas

#endif //TAPAS_HOT_TWOSIDE_INSP2_H_
