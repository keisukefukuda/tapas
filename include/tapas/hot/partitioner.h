#ifndef TAPAS_HOT_PARTITIONER__
#define TAPAS_HOT_PARTITIONER__

namespace tapas {
namespace hot {

template <class TSP> // Tapas static params
class Partitioner {
 private:
  const index_t max_nb_;

  using BodyType = typename TSP::Body;
  using BodyAttr = typename TSP::BodyAttr;
  using KeyType = typename Cell<TSP>::KeyType;
  using CellAttr = typename Cell<TSP>::CellAttr;
  using CellHashTable = typename Cell<TSP>::CellHashTable;

  using KeySet = typename Cell<TSP>::SFC::KeySet;

  using SFC = typename TSP::SFC;
  using HT = typename Cell<TSP>::CellHashTable;

  using Data = typename Cell<TSP>::Data;

 public:
  Partitioner(unsigned max_nb): max_nb_(max_nb) {}

  /**
   * @brief Partition the space and build the tree
   */
  Cell<TSP> *Partition(Data *data, const BodyType *b, const BodyAttr *a, const double *w, index_t nb, MPI_Comm comm);

 public:
  //---------------------
  // Supporting functions
  //---------------------

  /**
   * @brief Find owner process from a head-key list.
   * The argument head_list contains SFC keys that are the first keys of processes.
   * head_list[P] is the first SFC key belonging to process P.
   * Because the first element is always 0 (by definition of space filling curve),
   * the result must be always >= 0.
   *
   */
  static int
  FindOwnerProcess(const std::vector<KeyType> &head_list, KeyType key) {
    TAPAS_ASSERT(Cell<TSP>::SFC::RemoveDepth(head_list[0]) == 0);
    auto comp = [](KeyType a, KeyType b) {
      return Cell<TSP>::SFC::RemoveDepth(a) < Cell<TSP>::SFC::RemoveDepth(b);
    };
    return std::upper_bound(head_list.begin(), head_list.end(), key, comp) - head_list.begin() - 1;
  }

  static std::vector<int>
  FindOwnerProcess(const std::vector<KeyType> &head_key_list,
                   const std::vector<KeyType> &keys) {
    std::vector<int> owners(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
      owners[i] = FindOwnerProcess(head_key_list, keys[i]);
    }

    return owners;
  }

  /**
   * \brief Select cells to be sent as a response in Insp2::Exchange
   * The request lists are made conservatively, thus not all the requested cells exist in the sender process.
   * Check the requested list and replace non-existing cells with existing cells by the their finest anscestors.
   * If attribute of a cell is requested but the cell is actually a leaf,
   * both of the attribut and body must be sent.
   */
  static void SelectResponseCells(std::vector<KeyType> &cell_attr_keys, std::vector<int> &attr_src_pids,
                                  std::vector<KeyType> &leaf_keys, std::vector<int> &leaf_src_pids,
                                  const HT& hash) {
    std::set<std::pair<int, KeyType>> res_attr; // keys (and their destinations) of which attributes will be sent as response.
    std::set<std::pair<int, KeyType>> res_body; // keys (and their destinations) of which bodies will be sent as response.

    TAPAS_ASSERT(cell_attr_keys.size() == attr_src_pids.size());
    TAPAS_ASSERT(leaf_keys.size() == leaf_src_pids.size());

    for (size_t i = 0; i < cell_attr_keys.size(); i++) {
      KeyType k = cell_attr_keys[i];
      int src_pid = attr_src_pids[i]; // PID of the process that requested k.

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      if (k == 0) {
        // This should not happend because if k is root node,
        // that means the process does not have any anscestor of the original k (except the root).
        // The requester sent the request to a wrong process
        TAPAS_ASSERT(false);
      }

      res_attr.insert(std::make_pair(src_pid, k));

      if (hash.at(k)->IsLeaf()) {
        res_body.insert(std::make_pair(src_pid, k));
      }

    }

    for (size_t i = 0; i < leaf_keys.size(); i++) {
      KeyType k = leaf_keys[i];
      int src_pid = leaf_src_pids[i];

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      TAPAS_ASSERT(k != 0); // the same reason above
      TAPAS_ASSERT(hash.count(k) > 0);
      TAPAS_ASSERT(hash.at(k)->IsLeaf());

      res_body.insert(std::make_pair(src_pid, k));
    }

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&res_attr, &res_body](int rank, int) {
        std::cerr << "Rank " << rank << " SelectResponseCells: keys_attr.size() = " << res_attr.size() << std::endl;
        std::cerr << "Rank " << rank << " SelectResponseCells: keys.body.size() = " << res_body.size() << std::endl;
      });
#endif

    // Set values to the vectors
    cell_attr_keys.resize(res_attr.size());
    attr_src_pids.resize(res_attr.size());

    int idx = 0;
    for (auto & iter : res_attr) {
      attr_src_pids[idx] = iter.first;
      cell_attr_keys[idx] = iter.second;
      idx++;
    }

    leaf_keys.resize(res_body.size());
    leaf_src_pids.resize(res_body.size());

    idx = 0;
    for (auto & iter : res_body) {
      leaf_src_pids[idx] = iter.first;
      leaf_keys[idx] = iter.second;

      idx++;
    }

    return;
  }

  static void KeysToAttrs(const std::vector<KeyType> &keys,
                          std::vector<CellAttr> &attrs,
                          const HT& hash) {
    // functor
    auto key_to_attr = [&hash](KeyType k) -> CellAttr& {
      return hash.at(k)->attr();
    };
    
    attrs.resize(keys.size());
    std::transform(keys.begin(), keys.end(), attrs.begin(), key_to_attr);
  }

  static void KeysToBodies(const std::vector<KeyType> &keys,
                           std::vector<index_t> &nb,
                           std::vector<BodyType> &bodies,
                           const HT& hash) {
    nb.resize(keys.size());
    bodies.clear();

    // In BH, each leaf has 0 or 1 body (while every cell has attribute)
    for (size_t i = 0; i < keys.size(); i++) {
      KeyType k = keys[i];
      auto *c = hash.at(k);
      nb[i] = c->IsLeaf() ? c->nb() : 0;

      for (size_t bi = 0; bi < nb[i]; bi++) {
        bodies.push_back(c->body(bi));
      }
    }
  }

}; // class Partitioner

/**
 * @brief Partition the simulation space and build SFC key based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param numBodies Length of b (NOT the total number of bodies over all processes)
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 * @todo In this function keys are exchanged using alltoall communication, as well as bodies.
 *       In extremely large scale systems, calculating keys locally again after communication
 *       might be faster.
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename Cell<TSP>::Data *data,
                            const typename TSP::Body *b,
                            const typename TSP::BodyAttr *a,
                            const double *w,
                            index_t num_bodies,
                            MPI_Comm comm) {
  using SFC = typename TSP::SFC;
  using CellType = Cell<TSP>;
  using Data = typename CellType::Data;

  if (data == nullptr) {
    // First timestep
    data = new Data(comm);
    data->ncrit_ = max_nb_;
    data->sample_rate_ = SamplingOctree<TSP, SFC>::SamplingRate();
  } else {
    // if `data` is not NULL,
    // This is re-partitioning. Increase time step counter
    // for reporting.
    data->timestep_++;
  }

  // Build local trees
  SamplingOctree<TSP, SFC> stree(b, a, w, num_bodies, data, max_nb_);
  stree.Build();
  
  // Build Global trees
  GlobalTree<TSP>::Build(*data);

#if defined(TAPAS_TREE_STAT) || 1
  stree.ShowTreeStat();
#endif

#ifdef TAPAS_DEBUG_DUMP
  {
    tapas::debug::DebugStream e("cells");

    for (auto&& iter : data->ht_) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      e.out() << SFC::Simplify(k) << " "
              << "d=" << SFC::GetDepth(k) << " "
              << "leaf=" << c->IsLeaf() << " "
          //<< "owners=" << std::setw(2) << std::right << 0 << " "
              << "nb=" << std::setw(3) << (c->IsLeaf() ? (int)c->nb() : -1) << " "
              << "center=[" << c->center() << "] "
          //<< "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
          //<< "parent=" << SFC::Simplify(SFC::Parent(k))  << " "
              << std::endl;
    }
  }
#endif

  stree.ListShallowLeaves();

  // Initialize the mapper class (mainly for GPU)
  data->mapper_.Setup();

  // return the root cell (root key is always 0)
  return data->ht_[0];
}



} // ns hot 
} // ns tapas


#endif // TAPAS_HOT_PARTITIONER__
