#ifndef TAPAS_HOT_TARGET_SIDE_LET_H__
#define TAPAS_HOT_TARGET_SIDE_LET_H__

#include<vector>

#include <tapas/iterator.h>
#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/let_common.h>

#ifdef TAPAS_TWOSIDE_LET
#include <tapas/hot/inspector/twoside_on_target.h>
#else
#include <tapas/hot/inspector/oneside_on_target.h>
#include <tapas/hot/inspector/oneside_on_source.h>
#endif

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
struct TargetSideLET {
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

  using AttrTuple = std::tuple<KeyType, CellAttr>;
  using HT = typename CellType::CellHashTable;
  using SendKeys = std::unordered_map<int, std::set<KeyType>>; // map { destination rank => [keys] }

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

    MPI_Barrier(MPI_COMM_WORLD);
    bt_comm = MPI_Wtime();

    tapas::mpi::Alltoallv2(keys_attr_send, attr_dest, keys_attr_recv, attr_src, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(keys_body_send, body_dest, keys_body_recv, body_src, data.mpi_type_key_, MPI_COMM_WORLD);

    et_comm = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-req-comm", et_comm - bt_comm);

    et_all = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Map2-LET-req", et_all - bt_all);
  }

  /**
   * \brief Send response cells to each other
   */
  static std::tuple<std::vector<KeyType>, std::vector<CellAttr>, std::vector<AttrTuple>>
  ExchCells(Data &data,
            std::vector<KeyType> &keys,   /* in, out */
            std::vector<int> &dest_ranks,
            const SendKeys &amap){
    MPI_Barrier(data.mpi_comm_);
    double bt = MPI_Wtime();
    double bt_comp1 = MPI_Wtime();
    std::vector<AttrTuple> send_buf(keys.size());
    std::vector<int> send_count(data.mpi_size_);

    int prev_rank = dest_ranks[0];

    int size = data.mpi_size_;

    for (size_t i = 0; i < keys.size(); i++) {
      KeyType k = keys[i];
      int r = dest_ranks[i];

      // debug
      assert(prev_rank <= r);
      assert(r < data.mpi_size_);
      assert(data.ht_.count(k) > 0);

      send_buf[i] = std::make_tuple(k, data.ht_.at(k)->attr());
      send_count[r]++;
      prev_rank = r;
    }

    std::vector<AttrTuple> recv_buf;
    std::vector<int> recv_count;

    double et_comp1 = MPI_Wtime();
    
    MPI_Barrier(data.mpi_comm_);
    double bt_mpi = MPI_Wtime();
    tapas::mpi::Alltoallv(send_buf, send_count, recv_buf, recv_count, data.mpi_comm_);
    double et_mpi = MPI_Wtime();

    MPI_Barrier(data.mpi_comm_);
    double bt_comp2 = MPI_Wtime();
    std::vector<KeyType> res_keys(recv_buf.size());
    std::vector<CellAttr> res_attrs(recv_buf.size());
    //res_keys.reserve(recv_buf.size());
    //res_attrs.reserve(recv_buf.size());
    double et_comp2 = MPI_Wtime();

    MPI_Barrier(data.mpi_comm_);
    double bt_comp3 = MPI_Wtime();
    for (size_t i = 0; i < recv_buf.size(); i++) {
      res_keys[i] = std::get<0>(recv_buf[i]);
      res_attrs[i] = std::get<1>(recv_buf[i]);
    }
    double et_comp3 = MPI_Wtime();
    double et = MPI_Wtime();

    if (data.mpi_rank_ == 0) { std::cout << "debug: " << __FILE__ << ":" << __LINE__ << std::endl; }
#if 1
    tapas::debug::BarrierExec([&](int, int) {
        size_t count = send_buf.size();
        double size = count * sizeof(send_buf[0]);
        std::cout << "ExchCells: #cells = " << count << "  size=" << std::fixed << std::setprecision(2) << size
                  << "(=" << std::fixed << std::setprecision(2) << (size/1024/1024) << " MB)"
                  << std::endl;
        std::cout << "           ht_.size() = " << data.ht_.size() << std::endl;
      });

    tapas::debug::BarrierExec([&](int rank, int size) {
        std::cout << "ExchCell: [" << rank << "] send_count = ";
        for (int i : send_count) {
          std::cout << i << " ";
        }
        std::cout << std::endl;
        for (int r = 0; r < size; r++) {
          std::cout << amap.at(r).size() << " ";
        }
        std::cout << std::endl;
      });
#endif
    
    if (data.mpi_rank_ == 0) { std::cout << "debug: " << __FILE__ << ":" << __LINE__ << std::endl; }
    // if (data.mpi_rank_ == 0) {
    //   std::cout << "ExchCells: " << (et - bt) << " [s]" << std::endl;
    //   std::cout << "ExchCells: MPI: " << (et_mpi - bt_mpi) << " [s]" << std::endl;
    //   std::cout << "ExchCells: Pre1: " << (et_comp1 - bt_comp1) << " [s]" << std::endl;
    //   std::cout << "ExchCells: Pre2: " << (et_comp2 - bt_comp2) << " [s]" << std::endl;
    // }
    tapas::debug::BarrierExec([&](int rank, int) {
        if (rank == 0) {
          printf("%3s %7s %7s %7s %7s %7s %7s ExchCells\n", "rank", "Total", "MPI", "Comp1", "Comp2", "Comp3", "recvbuf.size()");
        }
        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f %10d ExchCells\n",
               rank,
               et-bt,
               et_mpi - bt_mpi,
               et_comp1 - bt_comp1,
               et_comp2 - bt_comp2,
               et_comp3 - bt_comp3,
               (int)recv_buf.size());
        // std::cout << "ExchCells: [" << rank << "] " << (et - bt) << " [s]" << std::endl;
        // std::cout << "ExchCells: [" << rank << "] MPI: " << (et_mpi - bt_mpi) << " [s]" << std::endl;
        // std::cout << "ExchCells: [" << rank << "] Pre1: " << (et_comp1 - bt_comp1) << " [s]" << std::endl;
        // std::cout << "ExchCells: [" << rank << "] Pre2: " << (et_comp2 - bt_comp2) << " [s]" << std::endl;
      });

    data.time_rec_.Record(data.timestep_, "Map2-LET-res-comp1", et_comp1 - bt_comp1);
    data.time_rec_.Record(data.timestep_, "Map2-LET-res-attr-comm", et - bt);
    return std::make_tuple(res_keys, res_attrs, recv_buf);
  }

  static void ExchBodies(Data &data, 
                         std::vector<KeyType> &req_leaf_keys,
                         std::vector<int> &leaf_src_ranks,
                         std::vector<CellAttr> &res_cell_attrs, std::vector<BodyType> &res_bodies,
                         std::vector<index_t> &res_nb){
    // Preapre all bodies to send to <leaf_src_ranks> processes
    double bt = MPI_Wtime();

    std::vector<int> leaf_dest = leaf_src_ranks;         // copy
    std::vector<KeyType> leaf_keys_sendbuf = req_leaf_keys; // copy
    res_bodies.clear();

    // First, leaf_keys_sendbuf must be ordered by thier destination processes
    // (Since we need to send bodies later, leaf_keys_sendbuf must NOT be sorted ever again.)
    tapas::SortByKeys(leaf_dest, leaf_keys_sendbuf);

    std::vector<index_t> leaf_nb_sendbuf (leaf_keys_sendbuf.size()); // Cell <leaf_keys_sendbuf[i]> has <leaf_nb_sendbuf[i]> bodies.
    std::vector<BodyType> body_sendbuf;

    std::vector<int> leaf_sendcnt(data.mpi_size_, 0); // used for <leaf_keys_sendbuf> and <leaf_nb_sendbuf>.
    std::vector<int> body_sendcnt(data.mpi_size_, 0);    // used for <bodies_sendbuf>

    for (size_t i = 0; i < leaf_keys_sendbuf.size(); i++) {
      KeyType k = leaf_keys_sendbuf[i];
      CellType *c = data.ht_.at(k);
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

#if 1 // performance measurement
    if (data.mpi_rank_ == 0) {
      size_t count = leaf_keys_sendbuf.size();
      double size = count * sizeof(leaf_keys_sendbuf[0]);
      std::cout << "ExchBodies: #leavess = " << count << "  size=" << std::fixed << size
                << "(=" << std::fixed << (size/1024/1024) << " MB)"
                << std::endl;
      std::cout << "           ht_.size() = " << data.ht_.size() << std::endl;
    }

    if (data.mpi_rank_ == 0) {
      size_t count = body_sendbuf.size();
      double size = count * sizeof(body_sendbuf[0]);
      std::cout << "ExchBodies: #bodies = " << count << "  size=" << std::fixed << size
                << "(=" << std::fixed << (size/1024/1024) << " MB)"
                << std::endl;
      std::cout << "           ht_.size() = " << data.ht_.size() << std::endl;
    }
#endif

    MPI_Barrier(data.mpi_comm_);
    double bt_mpi = MPI_Wtime();

    // Send response keys and bodies
    tapas::mpi::Alltoallv(leaf_keys_sendbuf, leaf_sendcnt, req_leaf_keys, leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(leaf_nb_sendbuf,   leaf_sendcnt, res_nb,        leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(body_sendbuf,      body_sendcnt, res_bodies,    body_recvcnt, MPI_COMM_WORLD);

    double et_mpi = MPI_Wtime();
    double et = MPI_Wtime();

    if (data.mpi_rank_ == 0) {
      std::cout << "ExchBodies: MPI: " << (et_mpi - bt_mpi) << " [s]" << std::endl;
      std::cout << "ExchBodies: " << (et - bt) << " [s]" << std::endl;
    }

    data.time_rec_.Record(data.timestep_, "Map2-LET-res-body-comm", et - bt);
  }

  /**
   * \brief Select cells to be sent as a response in Insp2::Exchange
   * The request lists are made conservatively, thus not all the requested cells exist in the sender process.
   * Check the requested list and replace non-existing cells with existing cells by the their finest anscestors.
   * If attribute of a cell is requested but the cell is actually a leaf,
   * both of the attribut and body must be sent.
   * 
   * Returns a SendKeyMap for CellAttrs
   */
  static std::tuple<SendKeys, SendKeys>
  SelectResponseCells(Data &data,
                      std::vector<KeyType> &cell_attr_keys,
                      std::vector<int> &attr_src_ranks,
                      std::vector<KeyType> &leaf_keys,
                      std::vector<int> &leaf_src_ranks,
                      const HT& hash) {
    std::set<std::pair<int, KeyType>> res_attr; // keys (and their destinations) of which attributes will be sent as response.
    std::set<std::pair<int, KeyType>> res_body; // keys (and their destinations) of which bodies will be sent as response.

    SendKeys amap; // keys to be sent and their destination ranks, of which CellAttr are sent.
    SendKeys bmap; // keys to be sent and their destination ranks, of which bodies are sent.

    TAPAS_ASSERT(cell_attr_keys.size() == attr_src_ranks.size());
    TAPAS_ASSERT(leaf_keys.size() == leaf_src_ranks.size());

    for (size_t i = 0; i < cell_attr_keys.size(); i++) {
      KeyType k = cell_attr_keys[i];
      int r = attr_src_ranks[i]; // rank that requested k.

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      if (k == 0) {
        // This should not happend because if k is root node,
        // that means the process does not have any anscestor of the original k (except the root).
        // The requester sent the request to a wrong process
        TAPAS_ASSERT(false);
      }

      res_attr.insert(std::make_pair(r, k));
      amap[r].insert(k);

      if (hash.at(k)->IsLeaf()) {
        res_body.insert(std::make_pair(r, k));
        bmap[r].insert(k);
      }
    }

    for (size_t i = 0; i < leaf_keys.size(); i++) {
      KeyType k = leaf_keys[i];
      int r = leaf_src_ranks[i];

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      TAPAS_ASSERT(k != 0); // the same reason above
      TAPAS_ASSERT(hash.count(k) > 0);
      TAPAS_ASSERT(hash.at(k)->IsLeaf());

      res_body.insert(std::make_pair(r, k));
      bmap[r].insert(k);
    }

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&res_attr, &res_body](int rank, int) {
        std::cerr << "Rank " << rank << " SelectResponseCells: keys_attr.size() = " << res_attr.size() << std::endl;
        std::cerr << "Rank " << rank << " SelectResponseCells: keys.body.size() = " << res_body.size() << std::endl;
      });
#endif

    // Set values to the vectors
    cell_attr_keys.resize(res_attr.size());
    attr_src_ranks.resize(res_attr.size());

    int idx = 0;
    for (auto & iter : res_attr) {
      attr_src_ranks[idx] = iter.first;
      cell_attr_keys[idx] = iter.second;
      idx++;
    }

    leaf_keys.resize(res_body.size());
    leaf_src_ranks.resize(res_body.size());

    idx = 0;
    for (auto & iter : res_body) {
      leaf_src_ranks[idx] = iter.first;
      leaf_keys[idx] = iter.second;

      idx++;
    }

    amap[data.mpi_rank_];
    bmap[data.mpi_rank_];
    return std::make_tuple(amap, bmap);
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
                       std::vector<CellAttr> &res_cell_attrs, std::vector<BodyType> &res_bodies,
                       std::vector<index_t> &res_nb){
    // req_attr_keys : list of cell keys of which cell attributes are requested
    // req_leaf_keys : list of cell keys of which bodies are requested
    // attr_src_ranks      : source process ranks of req_attr_keys (which are response target ranks)
    // leaf_src_ranks      : source process ranks of req_leaf_keys (which are response target ranks)

    // Code regions:
    //   1. Pre-comm computation
    //   2. Communication (Alltoallv)
    //   3. Post-comm computation
    double bt_all=0, et_all=0;

    using AttrTuple = std::tuple<KeyType, CellAttr>;
    std::vector<AttrTuple> recv_attrs;

    SendKeys amap, bmap;
    
    std::tie(amap, bmap) = SelectResponseCells(data, req_attr_keys, attr_src_ranks,
                                               req_leaf_keys, leaf_src_ranks,
                                               data.ht_);

    std::tie(req_attr_keys, res_cell_attrs, recv_attrs)
        = ExchCells(data, req_attr_keys, attr_src_ranks, amap);
    
    ExchBodies(data, req_leaf_keys, leaf_src_ranks, res_cell_attrs, res_bodies, res_nb);
    
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
   * \breif Register the received response cells to local LET hash table
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

    index_t body_offset = 0;
    for (size_t i = 0; i < res_leaf_keys.size(); i++) {
      KeyType k = res_leaf_keys[i];
      index_t nb = res_nb[i];
      index_t cur_body_offset = body_offset;
      body_offset += nb;

      if (data->ht_.count(k) > 0) {
        // received data already exists in local memory.
        // should be a warning?
        continue;
      }

      Cell<TSP> *c = nullptr;
      if (data->ht_let_.count(k) > 0) {
        // If the cell is already registered to ht_let_, the cell has attributes but not body info.
        c = data->ht_let_.at(k);
      } else if (data->ht_gtree_.count(k) > 0) {
        // it is a leaf in remote cell.
        c = data->ht_gtree_.at(k);
        if (!c->IsLeaf()) {
          c->is_local_ = false;
        } else {
          continue;
        }
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 1, data);
        data->ht_let_[k] = c;
      }

      c->is_leaf_ = true;
      c->nb_ = nb;
      c->bid_ = cur_body_offset;
    }

    double end = MPI_Wtime();
    data->time_rec_.Record(data->timestep_, "Map2-LET-register", end - beg);
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
    KeySet req_cell_attr_keys2; // cells of which attributes are to be transfered from remotes to local
    KeySet req_leaf_keys2; // cells of which bodies are to be transfered from remotes to local
    LetInspectorAction callback(data, req_cell_attr_keys, req_leaf_keys);
    LetInspectorAction callback2(data, req_cell_attr_keys2, req_leaf_keys2);

    double bt, et;

    // Depending on the macro, Tapas uses two-side or one-side inspector to construct LET.
    // One side traverse is much faster but it requires certain condition in user function f.
#ifdef TAPAS_TWOSIDE_LET
#warning "Using 2-sided LET"
    TwosideOnTarget<TSP> inspector(data);
    if (tapas::mpi::Rank() == 0) std::cout << "Using Target-side 2-sided LET" << std::endl;
#else
    OnesideOnTarget<TSP, UserFunct, Args...> inspector(data);
    if (tapas::mpi::Rank() == 0) std::cout << "Using Target-side 1-sided LET" << std::endl;
#endif

    bt = MPI_Wtime();
    inspector.Inspect(root, callback, f, args...);
    et = MPI_Wtime();

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
    Response(root.data(),
             res_cell_attr_keys, attr_src,
             res_leaf_keys, leaf_src, res_cell_attrs, res_bodies, res_nb);

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

#endif // TAPAS_HOT_TARGET_SIDE_LET_H__
