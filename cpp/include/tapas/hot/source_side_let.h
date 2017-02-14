#ifndef TAPAS_HOT_SOURCE_SIDE_LET_H__
#define TAPAS_HOT_SOURCE_SIDE_LET_H__

#include<vector>

#include <tapas/iterator.h>
#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/inspector/oneside_on_source.h>

using tapas::debug::BarrierExec;

namespace tapas {
namespace hot {

template<class TSP> class Cell;
template<class TSP> class Partitioner;

/**
 * A set of static functions to construct LET (Locally Essential Tree) in a target-side manner,
 * which means that target side processes run inspectors and request necessary data to 
 * owner (remote, source-side) processes.
 * 
 */
template<class TSP>
struct SourceSideLET {
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

  /**
   * Direction of Map-1 (Upward/Downward)
   */
  enum {
    MAP1_UP,
    MAP1_DOWN,
    MAP1_UNKNOWN
  };

  //using ProxyBody = tapas::hot::proxy::ProxyBody<TSP>;
  //using ProxyBodyAttr = tapas::hot::proxy::ProxyBodyAttr<TSP>;
#ifdef TAPAS_TWOSIDE_LET
  using Policy = tapas::hot::proxy::FullTraversePolicy<TSP>;
#else
  using Policy = tapas::hot::proxy::OnesideTraversePolicy<Dim, FP, Data>;
#endif

  using ProxyAttr = tapas::hot::proxy::ProxyAttr<TSP>;
  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, Policy>;
  using ProxyMapper = tapas::hot::proxy::ProxyMapper<TSP>;

  // Note for UserFunct template parameter:
  // The template parameter `UserFunct` is used between all the functions in LET class
  // and seems that it should be included in the class template parameter list along with TSP.
  // However, it is actually not possible because LET class is declared as 'friend' in the Cell class
  // only with TSP parameter. It's impossible to declare a partial specialization to be friend.

  static void ShowHistogram(const Data &data) {
    const int d = data.max_depth_;
    TAPAS_ASSERT(d <= SFC::MaxDepth());

#ifdef TAPAS_DEBUG
    const long ncells = data.ht_.size();
    const long nall   = (pow(8.0, d+1) - 1) / 7;
    BarrierExec([&](int,int) {
        std::cout << "Cells: " << ncells << std::endl;
        std::cout << "depth: " << d << std::endl;
        std::cout << "filling rate: " << ((double)ncells / nall) << std::endl;
      });
#endif

    std::vector<int> hist(d + 1, 0);
    for (auto p : data.ht_) {
      const auto *cell = p.second;
      if (cell->IsLeaf()) {
        hist[cell->depth()]++;
      }
    }

#ifdef TAPAS_DEBUG
    BarrierExec([&](int, int) {
        std::cout << "Depth histogram" << std::endl;
        for (int i = 0; i <= d; i++) {
          std::cout << i << " " << hist[i] << std::endl;
        }
      });
#endif
  }

  /**
   * \brief Send request to remote processes
   */
  static void Request(Data &data,
                      KeySet &req_keys_attr, KeySet &req_keys_body,
                      std::vector<KeyType> &keys_attr_recv,
                      std::vector<KeyType> &keys_body_recv,
                      std::vector<int> &attr_src,
                      std::vector<int> &body_src) {
    const auto &ht = data.ht_;
    double bt_all, et_all, bt_comm, et_comm;

    MPI_Barrier(MPI_COMM_WORLD);
    bt_all = MPI_Wtime();

    // return values
    keys_attr_recv.clear(); // keys of which attributes are requested
    keys_body_recv.clear(); // keys of which attributes are requested

    attr_src.clear(); // Process IDs that requested attr_keys_recv[i]
    body_src.clear(); // Process IDs that requested attr_body_recv[i]

    // Local cells don't need to be transfered.
    // FIXME: here we calculate difference of sets {necessary cells} - {local cells} in a naive way.
    auto orig_req_keys_attr = req_keys_attr;
    req_keys_attr.clear();
    for (auto &v : orig_req_keys_attr) {
      if (ht.count(v) == 0) {
        req_keys_attr.insert(v);
      }
    }

    auto orig_req_keys_body = req_keys_body;
    for (auto &v : orig_req_keys_body) {
      if (ht.count(v) == 0) {
        req_keys_body.insert(v);
      }
    }

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&](int rank, int) {
        std::cout << "rank " << rank << "  Local cells are filtered out" << std::endl;
        std::cout << "rank " << rank << "  req_keys_attr.size() = " << req_keys_attr.size() << std::endl;
        std::cout << "rank " << rank << "  req_keys_body.size() = " << req_keys_body.size() << std::endl;
        std::cout << std::endl;
      });

    BarrierExec([&](int rank, int) {
        if (rank == 0) {
          for (size_t i = 0; i < data.proc_first_keys_.size(); i++) {
            std::cerr << "first_key[" << i << "] = " << SFC::Decode(data.proc_first_keys_[i])
                      << std::endl;
          }
        }
      });
#endif

    // The root cell (key 0) is shared by all processes. Thus the root cell is never included in the send list.
    TAPAS_ASSERT(req_keys_attr.count(0) == 0);

    // Send cell request to each other
    // Transfer req_keys_attr using MPI_Alltoallv

    // Step 1 : Exchange requests

    // vectorized req_keys_attr. A list of cells (keys) that the local process requires.
    // (send buffer)
    std::vector<KeyType> keys_attr_send(req_keys_attr.begin(), req_keys_attr.end());
    std::vector<KeyType> keys_body_send(req_keys_body.begin(), req_keys_body.end());

    TAPAS_ASSERT((int)data.proc_first_keys_.size() == data.mpi_size_);

    // Determine the destination process of each cell request
    std::vector<int> attr_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_attr_send);
    std::vector<int> body_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_body_send);

    //MPI_Barrier(MPI_COMM_WORLD);
    bt_comm = MPI_Wtime();

    tapas::mpi::Alltoallv2(keys_attr_send, attr_dest, keys_attr_recv, attr_src, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(keys_body_send, body_dest, keys_body_recv, body_src, data.mpi_type_key_, MPI_COMM_WORLD);

    et_comm = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-req-comm", et_comm - bt_comm);

#ifdef TAPAS_DEBUG_DUMP
    {
      assert(keys_body_recv.size() == body_src.size());
      tapas::debug::DebugStream e("body_keys_recv");
      for (size_t i = 0; i < keys_body_recv.size(); i++) {
        e.out() << SFC::Decode(keys_body_recv[i]) << " from " << body_src[i] << std::endl;
      }
    }
#endif

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&](int rank, int) {
        std::cout << "rank " << rank << "  req_keys_attr.size() = " << req_keys_attr.size() << std::endl;
        std::cout << "rank " << rank << "  req_keys_body.size() = " << req_keys_body.size() << std::endl;
        std::cout << std::endl;
      });
#endif

    et_all = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-req", et_all - bt_all);
  }

  // Register received cells into the local hashtable (ht_let_)
  static void Register2(Data &data,
                        const std::vector<std::tuple<KeyType, CellAttr>> &recv_keys,
                        const std::vector<std::tuple<KeyType, int>> &recv_leaves,
                        const std::vector<std::tuple<BodyType, BodyAttrType>> &recv_bodies) {
    //using CTpl = std::tuple<KeyType, CellAttr>; // cell attr tuple
    //using LTpl = std::tuple<KeyType, int>;      // leave tuple
    using BTpl = std::tuple<BodyType, BodyAttrType>;

    double beg = MPI_Wtime();

    for (auto &&tpl : recv_keys) {
      KeyType k = std::get<0>(tpl);
      const CellAttr &attr = std::get<1>(tpl);

      if (data.ht_.count(k) > 0) {
        // This cell is already a local cell.
        continue;
      }

      Cell<TSP> *c = nullptr;

      if (data.ht_gtree_.count(k) > 0) {
        c = data.ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 0, &data);
      }
      c->attr() = attr;
      c->is_leaf_ = false;
      c->nb_ = 0;
      c->bid_ = 0;
      data.ht_let_[k] = c;
    }

    size_t body_ofst = data.let_bodies_.size();

    // copy body data to data.let_bodies_.
    std::transform(std::begin(recv_bodies), std::end(recv_bodies),
                   std::back_inserter(data.let_bodies_),
                   [](const BTpl &t) { return std::get<0>(t); });

    // copy body attr data to data.let_body_attrs_.
    std::transform(std::begin(recv_bodies), std::end(recv_bodies),
                   std::back_inserter(data.let_body_attrs_),
                   [](const BTpl &t) { return std::get<1>(t); });
    
    //size_t bo = 0; // body_ofst for debug
    // Register bodies
    for (auto &&tpl : recv_leaves) {
      KeyType k = std::get<0>(tpl);
      int nb = std::get<1>(tpl);
      Cell<TSP> *c = nullptr;

      if (data.ht_let_.count(k) > 0) {
        c = data.ht_let_.at(k);

        // // debug
        // std::cout << "Register2: " << SFC::Simplify(k) << " " << "isleaf=" << c->IsLeaf() << std::endl;
        // std::cout << "           " << c->nb() << " == " << nb << std::endl;
        // for (size_t ib = 0; ib < c->nb(); ib++, bo++) {
        //   std::cout << "            body " << ib  << std::endl;
        //   std::cout << "           " << c->body(ib).X << std::endl;
        //   std::cout << "           " << std::get<0>(recv_bodies[bo]).X << std::endl;
        //   std::cout << "           " << c->body_attr(ib) << std::endl;
        //   std::cout << "           " << std::get<1>(recv_bodies[bo]) << std::endl;
        //   std::cout << std::endl;
        // }
        
      } else if (data.ht_gtree_.count(k) > 0) {
        c = data.ht_gtree_.at(k);
        if (!c->IsLeaf()) {
          // if the 'already-local' cell is a non-leaf cell, update the cell information
          c->is_local_ = false;
        } else {
          // if it is a leaf, then the received cell is not necessary.
          body_ofst += nb;
          continue;
        }
      } else {
        // The received cell is not in local memory.
        c = Cell<TSP>::CreateRemoteCell(k, 1, &data);
        data.ht_let_[k] = c;
      }
      
      c->is_leaf_ = true;
      c->nb_ = nb;
      c->is_local_ = false;
      c->bid_ = body_ofst;
      body_ofst += nb;
    }
    
    double end = MPI_Wtime();

    if (tapas::mpi::Rank() == 0) {
      std::cout << "Register: " << (end - beg) << " [s]" << std::endl;
    }
  }  

  /**
   * Inspecting action for LET construction
   * TODO: LET用にセルをリスト追加するアクションのクラス
   */
  class LetInspectorAction {
    using HT = typename Cell<TSP>::CellHashTable;
    const HT &ht_;
    const HT &ht_gtree_;
    KeySet &attr_keys_; // cell keys
    KeySet &leaf_keys_; // leaf keys
   public:
    LetInspectorAction(const Data &data, KeySet &attr_keys, KeySet &leaf_keys)
        : ht_(data.ht_)
        , ht_gtree_(data.ht_gtree_)
        , attr_keys_(attr_keys)
        , leaf_keys_(leaf_keys)
    { }
    
    inline bool operator()(KeyType /* trg_key */, bool /* is_trg_leaf */,
                           KeyType src_key, bool is_src_leaf,
                           IntrFlag splt) {
      if (splt.IsReadAttrR() || splt.IsSplitR() || splt.IsApprox()) {
        attr_keys_.insert(src_key);
      }
      
      if (is_src_leaf) {
        leaf_keys_.insert(src_key);
      }

      return true;
    }
  };

  std::vector<std::tuple<KeyType, CellAttr>>
  static ExchCellAttrs(const Data &data, const std::unordered_map<int, KeySet> &attr_keys) {
    double bt = MPI_Wtime();
    using KATuple = std::tuple<KeyType, CellAttr>;
    std::vector<int> send_count(data.mpi_size_);
    std::vector<KATuple> send_buf;

    for (int rank = 0; rank < data.mpi_size_; rank++) {
      if (rank == data.mpi_rank_ || attr_keys.count(rank) == 0) { continue; }
      const auto &keys = attr_keys.at(rank);
      send_count[rank] = keys.size();
      
      int pos = send_buf.size();
      send_buf.resize(send_buf.size() + keys.size());
      for (KeyType k : keys) {
        TAPAS_ASSERT(data.ht_.count(k) > 0);
        send_buf[pos] = std::make_pair(k, data.ht_.at(k)->attr());
        pos++;
      }
    }

    double bt2 = MPI_Wtime();
    std::vector<KATuple> recv_buf;
    std::vector<int> recv_count;
    tapas::mpi::Alltoallv(send_buf, send_count, recv_buf, recv_count, data.mpi_comm_);
    double et2 = MPI_Wtime();

    double et = MPI_Wtime();
    if (data.mpi_rank_ == 0) {
      std::cout << "ExchCells: " << (et-bt) << " [s]" << std::endl;
      std::cout << "ExchCells: MPI: " << (et2-bt2) << " [s]" << std::endl;
    }
    return recv_buf;
  }

  /**
   *
   */
  std::tuple<std::vector<std::tuple<KeyType, int>>,
             std::vector<std::tuple<BodyType, BodyAttrType>>>
  static ExchBodies(const Data &data,  const std::unordered_map<int, KeySet> &leaf_keys) {
    double bt = MPI_Wtime();
    using KTuple = std::tuple<KeyType, int>; // pair of leaf key and the number of its bodies
    using BTuple = std::tuple<BodyType, BodyAttrType>;

    // leaf_keys = rank :: int -> keys :: set<KeyType>
    
    // first, exchange leaf keys and the number of bodies
    std::vector<int> send_count(data.mpi_size_), send_count_bodies(data.mpi_size_);
    std::vector<KTuple> send_buf;
    std::vector<BTuple> send_buf_bodies;

    for (int r = 0; r < data.mpi_size_; r++) {
      if (r == data.mpi_rank_ || leaf_keys.count(r) == 0) { continue; }
      const auto &keys = leaf_keys.at(r);
      int nb_rank_total = 0; // total number of bodies sent to rank r
      send_count[r] = keys.size();

      // pack bodies and body attributes into send_buf_bodies
      for (KeyType k: keys) {
        TAPAS_ASSERT(data.ht_.count(k) > 0 && data.ht_.at(k)->IsLeaf());
        const CellType &c = *(data.ht_.at(k));
        int nb = c.nb();
        send_buf.push_back(std::make_pair(k, nb));
        for (int i = 0; i < nb; i++) {
          send_buf_bodies.push_back(std::make_tuple(c.body(i), c.body_attr(i)));
        }
        nb_rank_total += nb;
      }
      send_count_bodies[r] = nb_rank_total;
    }

    std::vector<KTuple> recv_keys;
    std::vector<int>    recv_keys_cnt;
    std::vector<BTuple> recv_bodies;
    std::vector<int>    recv_bodies_cnt;

    double bt2 = MPI_Wtime();
    tapas::mpi::Alltoallv(send_buf, send_count, recv_keys, recv_keys_cnt, data.mpi_comm_);
    tapas::mpi::Alltoallv(send_buf_bodies, send_count_bodies, recv_bodies, recv_bodies_cnt, data.mpi_comm_);
    double et2 = MPI_Wtime();

    double et = MPI_Wtime();
    if (data.mpi_rank_ == 0) {
      std::cout << "ExchBodies: " << (et-bt) << " [s]" << std::endl;
      std::cout << "ExchBodies: MPI: " << (et2-bt2) << " [s]" << std::endl;
    }
    return std::make_tuple(recv_keys, recv_bodies);
  }
    
  /**
   * \brief Build Locally essential tree
   */
  template<class UserFunct, class...Args>
  static void Exchange(CellType &root, UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-All", SCOREP_USER_REGION_TYPE_FUNCTION);
    auto &data = root.data();
    double beg = MPI_Wtime();

#ifdef TAPAS_DEBUG_DUMP
    ShowHistogram(root.data());
#endif

    // Traverse
    KeySet req_cell_attr_keys; // cells of which attributes are to be transfered from remotes to local
    KeySet req_leaf_keys; // cells of which bodies are to be transfered from remotes to local
    KeySet send_attr_keys; // cells of which attributes are to be transfered from remotes to local
    KeySet send_leaf_keys; // cells of which bodies are to be transfered from remotes to local

    std::unordered_map<int, KeySet> send_attr_keys2;
    std::unordered_map<int, KeySet> send_leaf_keys2;
    
    LetInspectorAction callback(data, req_cell_attr_keys, req_leaf_keys);

    double bt, et;

    // Depending on the macro, Tapas uses two-side or one-side inspector to construct LET.
    // One side traverse is much faster but it requires certain condition in user function f.
    OnesideOnSource<TSP, UserFunct, Args...> inspector(data);
    if (tapas::mpi::Rank() == 0) std::cout << "Using Source-side LET" << std::endl;

    // Test source-side LET inspection
    send_attr_keys.clear();
    send_leaf_keys.clear();
    bt = MPI_Wtime();
    for (int r = 0; r < data.mpi_size_; r++) {
      KeySet &akeys = send_attr_keys2[r]; // implicitly initialize
      KeySet &lkeys = send_leaf_keys2[r];
      LetInspectorAction callback(data, akeys, lkeys);
      if (r != data.mpi_rank_) {
        inspector.Inspect(r, root.key(), callback, f, args...);
      }
    }
    et = MPI_Wtime();

    if (root.data().mpi_rank_ == 0) {
      std::cout << "Inspector2 : " << std::scientific << (et-bt) << " [s]" << std::endl;
    }

    std::vector<std::tuple<KeyType, CellAttr>>  recv_attrs;  // received keys and cell attributes
    std::vector<std::tuple<KeyType, int>>       recv_leaves; // keys of received bodies and numbers of bodies.
    std::vector<std::tuple<BodyType, BodyAttrType>> recv_bodies; // received bodies and body attributes

    // perform MPI communication and exchange necessary data
    recv_attrs = ExchCellAttrs(root.data(), send_attr_keys2);
    std::tie(recv_leaves, recv_bodies) = ExchBodies(root.data(), send_leaf_keys2);

    Register2(data, recv_attrs, recv_leaves, recv_bodies);

    double end = MPI_Wtime();
    root.data().time_rec_.Record(root.data().timestep_, "Map2-LET-all", end - beg);
  }

  static void DebugDumpCells(Data &data) {
    (void)data;
#ifdef TAPAS_DEBUG_DUMP
    // Debug
    // Dump all received cells to a file
    {
      tapas::debug::DebugStream e("cells_let");
      e.out() << "ht_let.size() = " << data.ht_let_.size() << std::endl;
      for (auto& iter : data.ht_let_) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        if (c == nullptr) {
          e.out() << "ERROR: " << SFC::Simplify(k) << " is NULL in hash LET." << std::endl;
        } else {
          e.out() << SFC::Simplify(k) << " "
                  << "d=" << SFC::GetDepth(k) << " "
                  << "leaf=" << c->IsLeaf() << " "
                  << "nb=" << std::setw(3) << (c->IsLeaf() ? tapas::debug::ToStr(c->nb()) : "N/A") << " "
                  << "center=[" << c->center() << "] "
                  << "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
                  << "parent=" << SFC::Simplify(SFC::Parent(k)) << " "
                  << std::endl;
        }
      }
    }
    
    {
      tapas::debug::DebugStream e("M_let");
      
      for (auto& iter : data.ht_let_) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        if (c == nullptr) {
          e.out() << "ERROR: " << SFC::Simplify(k) << " is NULL in hash LET." << std::endl;
        } else {
          e.out() << std::setw(20) << std::right << SFC::Simplify(c->key()) << " ";
          e.out() << std::setw(3) << c->depth() << " ";
          e.out() << (c->IsLeaf() ? "L" : "_") << " ";
          e.out() << c->attr().M << std::endl;
        }
      }
    }
#endif /* TAPAS_DEBUG_DUMP */
  }
};

} // namespace hot

} // namespace tapas

#endif // TAPAS_HOT_SOURCE_SIDE_LET_H__
