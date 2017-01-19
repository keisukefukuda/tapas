#ifndef TAPAS_HOT_EXACT_LET_
#define TAPAS_HOT_EXACT_LET_

#include<vector>

#include <tapas/iterator.h>
#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/let_common.h>
#include <tapas/hot/oneside_insp2.h>
#include <tapas/hot/twoside_insp2.h>

using tapas::debug::BarrierExec;

namespace tapas {
namespace hot {

template<class TSP> class Cell;
template<class TSP> class Partitioner;

/**
 * A set of static functions to construct LET (Locally Essential Tree)
 *
 * ExactInsp2 puts no assumption on user's function but has more overhead instead.
 * It emulates all the behavior of user's function.
 */
template<class TSP>
struct ExactInsp2 {
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
  using ProxyAttr = tapas::hot::proxy::ProxyAttr<TSP>;
  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;
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

    MPI_Barrier(MPI_COMM_WORLD);
    bt_comm = MPI_Wtime();

    tapas::mpi::Alltoallv2(keys_attr_send, attr_dest, keys_attr_recv, attr_src, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(keys_body_send, body_dest, keys_body_recv, body_src, data.mpi_type_key_, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
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


  /**
   * \brief Select cells and send response to the requesters.
   * \param data Data structure
   * \param [in,out] req_attr_keys Vector of SFC keys of cells of which attributes are sent in response
   * \param [in,out] attr_src      Vector of process ranks which requested req_attr_keys[i]
   * \param [in,out] req_leaf_keys Vector of SFC keys of leaf cells of which bodies are sent in response
   * \param [in,out] leaf_src      Vector of process ranks which requested req_leaf_keys[i]
   * \param [out] res_cell_attrs Vector of cell attributes which are recieved from remote ranks
   * \param [out] res_bodies     Vector of bodies which are received from remote ranks
   * \param [out] res_nb         Vector of number of bodies which res_cell_attrs[i] owns.
   *
   * \todo Parallelize operations
   */
  static void Response(Data &data,
                       std::vector<KeyType> &req_attr_keys, std::vector<int> &attr_src_ranks,
                       std::vector<KeyType> &req_leaf_keys, std::vector<int> &leaf_src_ranks,
                       std::vector<CellAttr> &res_cell_attrs, std::vector<BodyType> &res_bodies, std::vector<index_t> &res_nb){

    SCOREP_USER_REGION("LET-Response", SCOREP_USER_REGION_TYPE_FUNCTION);
    // req_attr_keys : list of cell keys of which cell attributes are requested
    // req_leaf_keys : list of cell keys of which bodies are requested
    // attr_src_ranks      : source process ranks of req_attr_keys (which are response target ranks)
    // leaf_src_ranks      : source process ranks of req_leaf_keys (which are response target ranks)

    // Code regions:
    //   1. Pre-comm computation
    //   2. Communication (Alltoallv)
    //   3. Post-comm computation
    double bt_all=0, et_all=0;
    double bt=0, et=0;

    // ===== Pre-comm computation =====
    // Create and send responses to the src processes of requests.

    MPI_Barrier(MPI_COMM_WORLD);
    bt_all = MPI_Wtime();

    Partitioner<TSP>::SelectResponseCells(req_attr_keys, attr_src_ranks,
                                          req_leaf_keys, leaf_src_ranks,
                                          data.ht_);

    const auto &ht = data.ht_;
    int mpi_size = data.mpi_size_;

    // Prepare cell attributes to send to <attr_src_ranks> processes
    std::vector<KeyType> attr_keys_send = req_attr_keys; // copy (split senbuf and recvbuf)
    std::vector<int> attr_dest_ranks = attr_src_ranks;
    res_cell_attrs.clear();
    std::vector<CellAttr> attr_sendbuf;
    Partitioner<TSP>::KeysToAttrs(attr_keys_send, attr_sendbuf, data.ht_);

    data.time_rec_.Record(data.timestep_, "Map2-LET-res-comp1", et - bt);

    // ===== 2. communication =====
    // Send response keys and attributes
    bt = MPI_Wtime();

    tapas::mpi::Alltoallv2(attr_keys_send, attr_dest_ranks, req_attr_keys,  attr_src_ranks, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(attr_sendbuf,   attr_dest_ranks, res_cell_attrs, attr_src_ranks, data.mpi_type_attr_, MPI_COMM_WORLD);

    et = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-res-attr-comm", et - bt);

#if 0
    // Dump request keys
    {
      tapas::debug::DebugStream e("attr_res_exa");
      for (size_t i = 0; i < attr_keys_send.size(); i++) {
        auto k = attr_keys_send[i];
        e.out() << SFC::Decode(k)
                << " (" << k << ")"
                << " to " << attr_dest_ranks[i] << std::endl;
      }
    }
#endif

    // ===== 3. Post-comm computation =====
    // Preapre all bodies to send to <leaf_src_ranks> processes
    // body_destは、いまのままalltoallvに渡すとエラーになる。
    // TODO: leaf_keys_send と body_dest と一緒にSortByKeysして（すでにされている？）、
    //       leaf_src_ranks を body_dest に書き換える必要がある（ループをまわす）

    std::vector<int> leaf_dest = leaf_src_ranks;         // copy
    std::vector<KeyType> leaf_keys_sendbuf = req_leaf_keys; // copy
    res_bodies.clear();

    // First, leaf_keys_sendbuf must be ordered by thier destination processes
    // (Since we need to send bodies later, leaf_keys_sendbuf must NOT be sorted ever again.)
    tapas::SortByKeys(leaf_dest, leaf_keys_sendbuf);

    std::vector<index_t> leaf_nb_sendbuf (leaf_keys_sendbuf.size()); // Cell <leaf_keys_sendbuf[i]> has <leaf_nb_sendbuf[i]> bodies.
    std::vector<BodyType> body_sendbuf;

    std::vector<int> leaf_sendcnt(mpi_size, 0); // used for <leaf_keys_sendbuf> and <leaf_nb_sendbuf>.
    std::vector<int> body_sendcnt(mpi_size, 0);    // used for <bodies_sendbuf>

    for (size_t i = 0; i < leaf_keys_sendbuf.size(); i++) {
      KeyType k = leaf_keys_sendbuf[i];
      CellType *c = ht.at(k);
      TAPAS_ASSERT(c->IsLeaf());
      leaf_nb_sendbuf[i] = c->nb();

      int dest = leaf_dest[i];
      leaf_sendcnt[dest]++;

      for (index_t bi = 0; bi < c->nb(); bi++) {
        body_sendbuf.push_back(c->body(bi));
        body_sendcnt[dest]++;
      }
    }

#ifdef TAPAS_DEBUG
    index_t nb_total  = std::accumulate(leaf_nb_sendbuf.begin(), leaf_nb_sendbuf.end(), 0);
    index_t nb_total2 = body_sendbuf.size();
    index_t nb_total3 = std::accumulate(body_sendcnt.begin(), body_sendcnt.end(), 0);

    TAPAS_ASSERT(nb_total  == nb_total2);
    TAPAS_ASSERT(nb_total2 == nb_total3);
#endif

    res_nb.clear();

    // This information is not necessary because source ranks of boides can be computed from
    // leaf_src_ranks_ranks and res_nb.
    std::vector<int> leaf_recvcnt; // we don't use this
    std::vector<int> body_recvcnt; // we don't use this

    MPI_Barrier(MPI_COMM_WORLD);
    bt = MPI_Wtime();

    // Send response keys and bodies
    tapas::mpi::Alltoallv(leaf_keys_sendbuf, leaf_sendcnt, req_leaf_keys, leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(leaf_nb_sendbuf,   leaf_sendcnt, res_nb,        leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(body_sendbuf,      body_sendcnt, res_bodies,    body_recvcnt, MPI_COMM_WORLD);

    et = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-res-body-comm", et - bt);

#ifdef TAPAS_DEBUG_DUMP
    tapas::debug::BarrierExec([&](int, int) {
        std::cout << "ht.size() = " << ht.size() << std::endl;
        std::cout << "req_attr_keys.size() = " << req_attr_keys.size() << std::endl;
        std::cout << "body_sendbuf.size() = " << body_sendbuf.size() << std::endl;
        std::cout << "local_bodies.size() = " << data.local_bodies_.size() << std::endl;
        std::cout << "res_bodies.size() = " << res_bodies.size() << std::endl;
      });
#endif

    // TODO: send body attributes
    // Now we assume body_attrs from remote process is all "0" data.

    data.let_bodies_.assign(std::begin(res_bodies), std::end(res_bodies));
    data.let_body_attrs_.resize(res_bodies.size());
    bzero(&data.let_body_attrs_[0], data.let_body_attrs_.size() * sizeof(data.let_body_attrs_[0]));

    TAPAS_ASSERT(data.let_bodies_.size() == res_bodies.size());

    et_all = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-res-all", et_all - bt_all);
  }

  /**
   * \breif Register response cells to local LET hash table
   * \param [in,out] data Data structure (cells are registered to data->ht_lt_)
   */
  static void Register(Data *data,
                       const std::vector<KeyType> &res_cell_attr_keys,
                       const std::vector<CellAttr> &res_cell_attrs,
                       const std::vector<KeyType> &res_leaf_keys,
                       const std::vector<index_t> &res_nb) {
    SCOREP_USER_REGION("LET-Register", SCOREP_USER_REGION_TYPE_FUNCTION);
    MPI_Barrier(MPI_COMM_WORLD);
    double beg = MPI_Wtime();

    // Register received LET cells to local ht_let_ hash table.
    for (size_t i = 0; i < res_cell_attr_keys.size(); i++) {
      KeyType k = res_cell_attr_keys[i];
      TAPAS_ASSERT(data->ht_.count(k) == 0); // Received cell must not exit in local hash.
      
      Cell<TSP> *c = nullptr;

      if (data->ht_gtree_.count(k) > 0) {
        c = data->ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 0, data);
      }
      c->attr() = res_cell_attrs[i];
      c->is_leaf_ = false;
      c->nb_ = 0;
      c->bid_ = 0;
      data->ht_let_[k] = c;
    }

    TAPAS_ASSERT(res_leaf_keys.size() == res_nb.size());

    // Register received leaf cells to local ht_let_ hash table.
    index_t body_offset = 0;
    for (size_t i = 0; i < res_leaf_keys.size(); i++) {
      KeyType k = res_leaf_keys[i];
      index_t nb = res_nb[i];
      Cell<TSP> *c = nullptr;

      if (data->ht_let_.count(k) > 0) {
        // If the cell is already registered to ht_let_, the cell has attributes but not body info.
        c = data->ht_let_.at(k);
      } else if (data->ht_gtree_.count(k) > 0) {
        c = data->ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 1, data);
        data->ht_let_[k] = c;
      }

      c->is_leaf_ = true;
      c->nb_ = nb;
      c->bid_ = body_offset;

      body_offset += nb;
    }

    double end = MPI_Wtime();
    data->time_rec_.Record(data->timestep_, "Map2-LET-register", end - beg);
  }

  /**
   * \brief Build Locally essential tree
   */
  template<class UserFunct, class...Args>
  static void Exchange(CellType &root, UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-All", SCOREP_USER_REGION_TYPE_FUNCTION);
    double beg = MPI_Wtime();

#ifdef TAPAS_DEBUG_DUMP
    ShowHistogram(root.data());
#endif

    // Traverse
    KeySet req_cell_attr_keys; // cells of which attributes are to be transfered from remotes to local
    KeySet req_leaf_keys; // cells of which bodies are to be transfered from remotes to local

    double bt = MPI_Wtime();

    // Depending on the macro, Tapas uses two-side or one-side inspector to construct LET.
    // One side traverse is much faster but it requires certain condition in user function f.
#ifdef TAPAS_TWOSIDE_LET
#warning "Using 2-sided LET"
    TwosideInsp2<TSP>::Inspect(root, req_cell_attr_keys, req_leaf_keys, f, args...);
#else
    OnesideInsp2<TSP>::Inspect(root, req_cell_attr_keys, req_leaf_keys, f, args...);
#endif
    
    double et = MPI_Wtime();
    if (root.data().mpi_rank_ == 0) {
      std::cout << "Inspector : " << std::scientific << (et-bt) << " [s]" << std::endl;
    }

    req_cell_attr_keys.insert(req_leaf_keys.begin(), req_leaf_keys.end());

    // We need to convert the sets to vectors
    std::vector<KeyType> res_cell_attr_keys; // cell keys of which attributes are requested
    std::vector<KeyType> res_leaf_keys; // leaf cell keys of which bodies are requested

    std::vector<int> attr_src; // Process IDs that requested attr_keys_recv[i] (output from Request())
    std::vector<int> leaf_src; // Process IDs that requested attr_body_recv[i] (output from Request())

    // Request
    Request(root.data(), req_cell_attr_keys, req_leaf_keys,
            res_cell_attr_keys, res_leaf_keys, attr_src, leaf_src);

    // Response
    std::vector<CellAttr> res_cell_attrs;
    std::vector<BodyType> res_bodies;
    std::vector<index_t> res_nb; // number of bodies responded from remote processes
    Response(root.data(), res_cell_attr_keys, attr_src, res_leaf_keys, leaf_src, res_cell_attrs, res_bodies, res_nb);

    // Register
    Register(root.data_, res_cell_attr_keys, res_cell_attrs, res_leaf_keys, res_nb);

#ifdef TAPAS_DEBUG_DUMP
    DebugDumpCells(root.data());
#endif

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

#endif // TAPAS_HOT_EXACT_LET_
