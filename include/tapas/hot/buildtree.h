/** \file
 *  Functions to construct HOT octrees by a sampling method
 */

#ifndef TAPAS_HOT_BUILDTREE_H
#define TAPAS_HOT_BUILDTREE_H

#include "tapas/stdcbug.h"

#include <vector>
#include <iterator>
#include <algorithm>

#include <mpi.h>

#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/mpi_util.h"
#include "tapas/hot/global_tree.h"

#ifdef __CUDACC__
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif

namespace tapas {
namespace hot {

// Sampling rate
#ifndef TAPAS_SAMPLING_RATE
# define TAPAS_SAMPLING_RATE (1e-2)
#endif

template<class TSP, class SFC> struct SharedData;
template<class TSP> class Cell;


/**
 * \class SamplingOctree
 * \brief Collection of static functions for sampling-based octree construction.
 *
 */
template<class TSP, class SFC_>
class SamplingOctree {
 public:
  static const constexpr int kDim = TSP::Dim;
  static const constexpr int kPosOffset = TSP::kBodyCoordOffset;

  using FP = typename TSP::FP;
  using SFC = SFC_;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using KeySet = std::unordered_set<KeyType>;
  using BodyType = typename TSP::Body;
  using BodyAttrType = typename TSP::BodyAttr;
  using Data = SharedData<TSP, SFC>;
  //template<class T> using Allocator = typename Data::template Allocator<T>;
  template<class T> using Allocator = std::allocator<T>;
  using Reg = tapas::Region<kDim, FP>;

  template<class T> using Vector = std::vector<T, Allocator<T>>;

 private:
  std::vector<BodyType> bodies_; // bodies
  std::vector<BodyAttrType> attrs_;  // body attributes
  std::vector<double> weights_;  // weights of bodies
  std::vector<KeyType> body_keys_;
  std::vector<KeyType> proc_first_keys_; // first key of each process's region
  Reg region_;
  Data* data_;
  int ncrit_;

  static std::vector<BodyAttrType> InitAttr(const BodyAttrType *a, index_t nb) {
    std::vector<BodyAttrType> v;
    if (a == nullptr) {
      v.resize(nb);
      memset(v.data(), 0, sizeof(attrs_[0]) * nb);
    } else {
      v.assign(a, a + nb);
    }
    return v;
  }

  static std::vector<BodyType> InitBody(const BodyType *b, index_t nb) {
    if (b == nullptr) {
      std::cerr << "Body pointer must not be NULL." << std::endl;
      TAPAS_ASSERT(0);
    }
    return std::vector<BodyType>(b, b + nb);
  }

 public:
  SamplingOctree(const BodyType *b, const BodyAttrType *a, const double *w, index_t nb, Data *data, int ncrit)
      : bodies_(InitBody(b, nb))
      , attrs_ (InitAttr(a, nb))
      , weights_()
      , body_keys_(), proc_first_keys_(), region_(), data_(data), ncrit_(ncrit)
  {
    Vec<kDim, FP> local_max, local_min;
    
    for (index_t i = 0; i < nb; i++) {
      Vec<kDim, FP> pos = ParticlePosOffset<kDim, FP, kPosOffset>::vec(reinterpret_cast<const void*>(b+i));
      for (int d = 0; d < kDim; d++) {
        local_max[d] = (i == 0) ? pos[d] : std::max(pos[d], local_max[d]);
        local_min[d] = (i == 0) ? pos[d] : std::min(pos[d], local_min[d]);
      }
    }

    // If nullptr is specified as weights, use a vector of 1.0
    if (w == nullptr) {
      weights_.resize(nb, 1.0);
    } else {
      weights_.assign(w, w + nb);
    }

    region_.min() = local_min;
    region_.max() = local_max;

  }

  template<class MapType>
  void CountLocalCells(KeyType k, MapType &m) {
    int d = SFC::GetDepth(k);
    m[d] += 1;

    if (!data_->ht_.at(k)->IsLeaf()) {
      for (KeyType ch: SFC::GetChildren(k)) {
        CountLocalCells(ch, m);
      }
    }
  }
  
  template<class MapType>
  void CountGlobalCells(KeyType k, MapType &m) {
    if (data_->ht_gtree_.count(k) > 0) {
      int d = SFC::GetDepth(k);
      m[d]++;

      if (!data_->ht_gtree_[k]->IsLeaf()) { // if not leaf
        for (KeyType ch: SFC::GetChildren(k)) {
          CountGlobalCells(ch, m);
        }
      }
    }
  }

  /**
   * \brief Show tree statistics
   *
   * - Filling factor of tree
   */
  void ShowTreeStat() {
    using tapas::debug::BarrierExec;

    // calculate filling rate of each local trees,
    // and aggregate the results to the root process

    std::vector<unsigned long> cnt(data_->max_depth_ + 1); // cell count per depth

    // Count up local cell (cells under local trees)
    for (KeyType lroot : data_->lroots_) {
      // count cells per depth for each local root
      if (!data_->ht_[lroot]->IsLeaf()) {
        // do not count local root (because a local root is a global leaf as well)
        for (KeyType ch : SFC::GetChildren(lroot)) {
          CountLocalCells(ch, cnt);
        }
      }
    }

    // Aggregate the counted values to rank 0
    std::vector<unsigned long> recv_buf;
    mpi::Reduce(cnt, recv_buf, MPI_SUM, 0, MPI_COMM_WORLD);

    if (tapas::mpi::Rank() == 0) {
      // count cells in the global tree
      CountGlobalCells(0, recv_buf);
      double num = 0, den = 0; // numerator, denominator

      for (int d = 0; d < data_->max_depth_; d++) {
        double fr = (double) recv_buf[d] / pow(pow(2,kDim), d); // fill rate
        num += recv_buf[d];
        den += pow(pow(2,kDim), d);
        printf("%2d %3ld %.5f\n", d, recv_buf[d], fr);
      }

      std::cout << "Total Filling Rate : " << (num/den) << std::endl;
    }
  }

  /**
   * Calculate the region of the whole space by exchanging max/min coordinates of the bodies.
   */
  void ExchangeRegion() {
    Vec<kDim, FP> new_max, new_min;

    // Exchange max
    tapas::mpi::Allreduce(&region_.max()[0], &new_max[0], kDim,
                          MPI_MAX, MPI_COMM_WORLD);

    // Exchange min
    tapas::mpi::Allreduce(&region_.min()[0], &new_min[0], kDim,
                          MPI_MIN, MPI_COMM_WORLD);

    region_ = Reg(new_min, new_max);
  }

  /**
   * \brief Split the finest-level keys into `mpi_size` groups. Used in the DD-process.
   *
   * \param keys Keys of sampled bodies
   * \param mpi_size Number of processes
   *
   * Split L-level keys into `mpi_size` groups.
   *
   * We determine L by
   *     B = 2^Dim  (B = 8 in 3 dim space)
   *     Np = mpi_size
   *     Ls = log_B(Np) + 2
   *     (Ls is a starting value of L)
   *
   * L must be larger than the number of processes, and large enough to achieve good load balancing.
   * However, too large L leads to unnecessary deep tree structure because domain boundary may be too
   * close to a certain particle.
   */
  static std::vector<KeyType> PartitionSpace(const std::vector<KeyType> &body_keys,
                                             const std::vector<double> &body_weights,
                                             std::vector<double> &proc_weights,
                                             int mpi_size) {

    if (mpi_size == 1) {
      // If single process, there's no need to space partitioning. The rank 0 owns all.
      std::vector<KeyType> beg_keys = {0};
      return beg_keys;
    }
    
    // total weight
    const double totalw = std::accumulate(body_weights.begin(), body_weights.end(), 0);

    // Beginning key of each process.
    // This is the target value of this function to be returned to the caller.
    std::vector<KeyType> beg_keys(mpi_size);
    proc_weights.clear();
    proc_weights.resize(mpi_size, 0);

#if 0 // new code; suspended until the new source-side inspector is implemented.
    const double quota = totalw / mpi_size;
    // new code

    std::cout << "mpi_size = " << mpi_size << std::endl;
    std::cout << "total weight = " << totalw << std::endl;
    std::cout << "quota = " << quota << std::endl;

    size_t bi = 0; // body index
    size_t pi = 0; // proc index
    beg_keys[pi] = 0;
    for (pi = 0; pi < (size_t)mpi_size - 1; pi++) { // pi = process index
      while (proc_weights[pi] < quota) {
        proc_weights[pi] += body_weights[bi];
        beg_keys[pi + 1] = body_keys[bi];
        bi++;
      }
    }
    // pi == mpi_size - 1
    beg_keys[pi] = body_keys[bi];
    
    while(bi < body_keys.size()) {
      proc_weights[pi] += body_weights[bi++];
    }

    for (size_t i = 0; i < proc_weights.size(); i++) {
      std::cout << proc_weights[i] << " " << beg_keys[i] << std::endl;
    }

    TAPAS_ASSERT(beg_keys[0] == 0);

    return beg_keys;
    
#else // ----------- old code
    
    const int B = 1 << kDim; // 8 in 3-dim space
    const int Ls  = (int)(log((double)mpi_size) / log((double)B) + 2); // logB(Np) = log(Np) / log(B)
    
    for (int L = Ls; L < SFC::MaxDepth(); L++) {
      //double t = MPI_Wtime();
      //L = SFC::MaxDepth() - 1; // debug
      // Loop over [Ls, Ls+1, ...] until the load balancing seems good.
      
      // The value 'Ls + 5' is hardcoded.
      // If the value is too large, old-inspector takes longer
      // (because the hypothetical global tree gets higher)
      
      const KeyType K0 = SFC::AppendDepth(0, L); // the first key in level L
      const long W = pow(B, L); // number of cells in level L
      TAPAS_ASSERT(W > 0); // detect overflow

      //std::cout << "L=" << L << std::endl;
      if (W <= mpi_size) {
        std::cerr << "W=" << W << ", mpi_size=" << mpi_size << std::endl;
      }
      TAPAS_ASSERT(W > mpi_size); (void)W;
      
#if 0 // debug print: to be removed
      std::cerr << "mpi_size = " << mpi_size << std::endl;
      std::cerr << "B = " << B << std::endl;
      std::cerr << "L = " << L << std::endl;
      std::cerr << "W = " << W << std::endl;
#endif
      
      // Scan over the weight vector and find beginning keys.
      KeyType k = K0; // current key
      int ki = 0; // key index (to avoid overrun)
      double took_weight = 0; // sum of weight that is already taken former processes [0, pi)
      
      beg_keys[0] = K0; // The beginning key of the first process is always the first key of level L.

      for (int pi = 1; pi < mpi_size; pi++) {
        // Target weight of process pi
        double trg_w = (totalw - took_weight) / (mpi_size - pi + 1);
        
        //std::cout << "pi=" << (pi-1) << " target weight=" << trg_w << std::endl;
        
        // pi = process index
        double w = 0;  // weight of the *previous* process
        
        // Find the range of `k` from `body_keys` and add the range weight to `proc_weight`
        index_t b = 0, e = 0;
        for (; w < trg_w; k = SFC::GetNext(k), ki++) {
          assert(ki < W); // something is wrong. ki should look over the range of (0, W]
          
          // Find the range of bodies that belongs to `k`
          SFC::FindRangeByKey(body_keys, k, b, e);
          
          // sum of the body weights in the range [b,e)
          w += accumulate(body_weights.begin() + b, body_weights.begin() + e, 0);
        }
        //std::cout << std::endl;

        
        beg_keys[pi] = k;       // k is the key of the process
        proc_weights[pi-1] = w; // weight of the *previous* process
        took_weight += w;
        
        if (pi == mpi_size - 1) {
          // if `pi` is the last process, compute the weight of itself
          proc_weights[pi] = accumulate(body_weights.begin() + e, body_weights.end(), 0);
        }
      }

      // compute the stddev of weights and check it's acceptable, increase L if not.
      double mean = std::accumulate(std::begin(proc_weights),
                                    std::end(proc_weights),
                                    0) / mpi_size;
      double sigma = tapas::util::stddev(proc_weights); // standard deviation
      double ratio = sigma / mean;

#if 0 // debug outputs: to be removed.
      const double q = totalw / mpi_size; // quota: each process should have roughly totalw/mpi_size weight

      std::cout << "--------------------" << std::endl;
      std::cout << "L=" << L << std::endl;
      // std::cout << "body weights = ";
      // for (auto w : body_weights) std::cout << (int)w << " ";
      // std::cout << std::endl;
        
      std::cout << "total weights = " << (int)totalw << std::endl;
      std::cout << "q = " << q << std::endl;
        
      std::cout << "proc_weights = ";
      for (auto w : proc_weights) std::cout << (int)w << " ";
      std::cout << std::endl;

      std::cout << "MEAN   = " << mean << std::endl;
      std::cout << "STDDEV = " << sigma << std::endl;
      
      for (int i = 0; i < mpi_size; i++) {
        std::cout << i << " " << SFC::Decode(beg_keys[i]) << std::endl;
      }
      
      std::cout << "Ratio = " << ratio << std::endl;
#endif

      if (ratio < 0.05) break;

      // to be removed.
      else {
        if (L == SFC::MaxDepth() - 1) {
          std::cerr << "Failed to find smaller L. using the max L = " << L << std::endl;
        }
      }
    } // end for : if the load imbalance si less than 1%

#endif // old code

    return beg_keys;
  }

  /**
   * \brief Get sampling rate configuration
   */
  static double SamplingRate() {
    double R = 0.005;

#ifdef TAPAS_SAMPLING_RATE
    R = (TAPAS_SAMPLING_RATE);
#endif

    if (getenv("TAPAS_SAMPLING_RATE")) {
      R = atof(getenv("TAPAS_SAMPLING_RATE"));
    }
    
    TAPAS_ASSERT(0.0 < R && R < 1.0);

    return R;
  }

  /**
   * \brief Sample bodies and determine proc_first_keys_.
   * Output: proc_first_keys_.
   */
  void Sample() {
    const double R = SamplingRate();
    double beg = MPI_Wtime();

    data_->sampling_rate = R;

    // todo:
    // record R

    ExchangeRegion();
    data_->region_ = region_;

    int min_sample_nb = std::min((int)100, (int)bodies_.size());

    // sample bodies in this process
    int sample_nb = std::max((int)(bodies_.size() * R),
                             (int)min_sample_nb);

    std::vector<BodyType> sb(sample_nb); // local sampled bodies
    std::vector<double> sw(sample_nb);   // corresponding weights to sampled bodies

    assert(bodies_.size() == weights_.size());

    // Sample bodies by strided access
    int stride = bodies_.size() / sample_nb;
    for (int i = 0; i < sample_nb; i++) {
      sb[i] = bodies_[i * stride];
      sw[i] = weights_[i * stride];
    }

    std::vector<KeyType> sk = BodiesToKeys(sb, data_->region_); // keys of local sampled bodies
    std::vector<KeyType> sampled_keys;    // gather()ed sampled body keys
    std::vector<double> sampled_weights;  // gather()ed sampled weights

    // Gather the sampled particles into the DD-process
    int dd_proc_id = DDProcId();

    // Gather the sampled keys and weights
    tapas::mpi::Gatherv(sk, sampled_keys, dd_proc_id, MPI_COMM_WORLD);
    tapas::mpi::Gatherv(sw, sampled_weights, dd_proc_id, MPI_COMM_WORLD);

    // check
#ifdef TAPAS_DEBUG
    std::unordered_map<KeyType, double> w;
    for (size_t i = 0; i < sampled_keys.size(); i++) {
      w[sampled_keys[i]] = sampled_weights[i];
    }
#endif

    // Sort the body keys and corresponding weights
#ifdef __CUDACC__
    tapas::util::TiedSort2<KeyType, double>(sampled_keys, sampled_weights);
#else
    tapas::util::TiedSort<KeyType, double>(sampled_keys, sampled_weights);
#endif

#ifdef TAPAS_DEBUG
    // check
    for (size_t i = 0; i < sampled_keys.size(); i++) {
      assert(w[sampled_keys[i]] == sampled_weights[i]);
    }
#endif

    proc_first_keys_.resize(data_->mpi_size_);
    std::vector<double> proc_weights(data_->mpi_size_);

    if (data_->mpi_rank_ == dd_proc_id) { // in DD-process
      // Sampled keys are body keys, so the depth must be SFC::MAX_DEPTH
      TAPAS_ASSERT(SFC::GetDepth(sampled_keys[0]) == SFC::MAX_DEPTH);

      proc_first_keys_ = PartitionSpace(sampled_keys, sampled_weights, proc_weights, data_->mpi_size_);
      TAPAS_ASSERT((int)proc_first_keys_.size() == data_->mpi_size_);
    }

    // Each process's starting key is broadcast.
    TAPAS_ASSERT((int)proc_first_keys_.size() == data_->mpi_size_);
    tapas::mpi::Bcast(proc_first_keys_, dd_proc_id, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    data_->time_rec_.Record(data_->timestep_, "Tree-sample", end - beg);
  }

  /**
   * \brief Exchange bodies to owner processes determined by Sample() function.
   */
  void Exchange() {
    double beg = MPI_Wtime();

    data_->nb_before = bodies_.size();

    // Exchange bodies according to proc_first_keys_
    std::tie(bodies_, attrs_) = ExchangeBodies(bodies_, attrs_, proc_first_keys_, region_, data_->mpi_comm_); 
    body_keys_ = BodiesToKeys(bodies_, region_);

    // Sort both new_keys and new_bodies.
    SortByKeys(body_keys_, bodies_, attrs_);

    // todo: record bodies

    data_->local_bodies_.assign(bodies_.begin(), bodies_.end());
    data_->local_body_attrs_.assign(attrs_.begin(), attrs_.end());
    data_->local_body_keys_ = body_keys_;

    //data_->local_body_attrs_.resize(bodies_.size());
    //bzero(data_->local_body_attrs_.data(), sizeof(BodyAttrType) * bodies_.size());
    data_->nb_after = data_->local_bodies_.size();

    double end = MPI_Wtime();
    data_->time_rec_.Record(data_->timestep_, "Tree-exchange", end - beg);
    data_->time_rec_.Record(data_->timestep_, "Bodies", data_->local_bodies_.size());
  }

  /**
   * \brief Build an octree from bodies b, with a sampling-based method
   */
  void Build() {
    double beg = MPI_Wtime();

    index_t nb_total = 0;
    tapas::mpi::Allreduce((index_t)bodies_.size(), nb_total, MPI_SUM, MPI_COMM_WORLD);
    data_->nb_total = nb_total;

    Sample();

    Exchange();

    GrowLocal();

    // Get the max depth
    int d = data_->max_depth_;
    tapas::mpi::Allreduce(&d, &data_->max_depth_, 1, MPI_MAX, MPI_COMM_WORLD);
    data_->proc_first_keys_ = std::move(proc_first_keys_);

    // error check
    if (data_->ht_[0] == nullptr) {
      // If no leaf is assigned to the process, root node is not generated
      if (data_->mpi_rank_ == 0) {
        std::cerr << "There are too few particles compared to the number of processes."
                  << std::endl;
      }
      MPI_Finalize();
      exit(-1);
    }

    data_->nleaves = data_->leaf_keys_.size();
    data_->ncells = data_->ht_.size();

    double end = MPI_Wtime();
    data_->time_rec_.Record(data_->timestep_, "Tree-all", end - beg);

  }

  /**
   * \brief Grow the local tree, from local bodies, leaves to the root cell
   */
  void GrowLocal() {
    double beg = MPI_Wtime();
    proc_first_keys_.push_back(SFC::GetNext(0));

    GenerateCell((KeyType)0, std::begin(body_keys_), std::end(body_keys_));

    TAPAS_ASSERT(data_->ht_[0]->local_nb() == body_keys_.size());
    proc_first_keys_.pop_back();

    double end = MPI_Wtime();
    data_->time_rec_.Record(data_->timestep_, "Tree-growlocal", end - beg);
  }

  /**
   * Generate a cell object of Key k if it is within the range of local bodies
   */
  void GenerateCell(KeyType k,
                    typename std::vector<KeyType>::const_iterator pbeg, // beg of a subset of bkeys
                    typename std::vector<KeyType>::const_iterator pend // end of a subset of bkeys
                    ) {
    KeyType k2 = SFC::GetNext(k);
    auto bbeg = body_keys_.begin();

    int rank = data_->mpi_rank_;

    // find the range of bodies that belong to the cell k by binary searching.
    auto range_beg = std::lower_bound(pbeg, pend, k);
    auto range_end = std::lower_bound(pbeg, pend, k2);
    int nb = range_end - range_beg;

    // print bodies in the leaf
    // for (auto i = range_beg; i < range_end; i++) {
      
    // }

    // Checks if the cell is (completely) included in the process or strides over two processes.
    // If the cell strides over multiple processes, it's never a leaf and must be split, even if nb <= ncrit.
    bool included = SFC::Includes(proc_first_keys_[rank], proc_first_keys_[rank+1], k);

    // if (nb <= ncrit) and this process owns the cell (i.e. included == true, no other process owns any descendants of the cell),
    // the cell is a leaf.
    bool is_leaf = (nb <= ncrit_) && included;
    int body_beg = is_leaf ? range_beg - bbeg : 0;

    // Construct a cell.
    //auto reg = CellType::CalcRegion(k, data_->region_);
    CellType *c = new CellType(k, data_->region_, body_beg, nb);
    //c->key_ = k;
    c->is_leaf_ = is_leaf;
    c->is_local_ = true;
    c->is_local_subtree_ = false;
    c->nb_ = nb;
    c->local_nb_ = nb;
    c->data_ = data_;
    c->bid_ = body_beg;

    TAPAS_ASSERT(nb >= 0);
    TAPAS_ASSERT(body_beg >= 0);

    data_->ht_[k] = c;

    if (SFC::GetDepth(k) > data_->max_depth_) {
      data_->max_depth_ = SFC::GetDepth(k);
    }
    TAPAS_ASSERT(SFC::GetDepth(k) <= SFC::MaxDepth() &&
                 data_->max_depth_ <= SFC::MaxDepth());

    if (is_leaf) {
      // The cell [k] is a leaf.
      data_->leaf_keys_.push_back(k);
      data_->leaf_nb_.push_back(nb);
      // todo remove Data::leaf_owners_
    } else {
      if (SFC::GetDepth(k) == SFC::MaxDepth()) {
        // FATAL: the depth reached the maximum. We need to use larger type for KeyType.
        std::cerr << "FATAL: The depth of the tree reached the maximum (" << SFC::GetDepth(k)  << ") of Morton keys.\n"
                  << "       There are " << nb << " bodies in the leaf\n"
                  << "       Suggestion: \n"
                  << "         * Use larger max_nb (an argument to Partition())\n"
                  << "         * Use larger key type (an template argument to tapas::HOT).\n"
                  << std::endl;
        exit(-1);
      }
      
      // The cell [k] is not a leaf. Split it again.
      // Note: if the cell is not a leaf and nb == 0, that means other processes may have particles which belong to the cell.
      auto ch_keys = SFC::GetChildren(k);

      for (auto chk : ch_keys) {
        // Generate the child cell recursively  if the child key is in the range of this process,
        // otherwise ignore it.
        bool overlap = SFC::Overlapped(chk, SFC::GetNext(chk),
                                       proc_first_keys_[rank],
                                       proc_first_keys_[rank+1]);
        
        // Note: SFC::GetNext(0) is the next key of the root key 0, which means
        //       the `end` of the whole region

        if (overlap) {
          GenerateCell(chk, range_beg, range_end);
        }
      }
    }
  }

  void ListShallowLeaves() {
    std::vector<KeyType> sendbuf, recvbuf;
    for (KeyType k : data_->lroots_) {
      FindShallowLeaves(k, k, sendbuf); // List up `only child` cells for branch pruning in LET construction phase.
    }

    tapas::mpi::Allgatherv(sendbuf, recvbuf, data_->mpi_comm_);

    for (KeyType k : recvbuf) {
      data_->shallow_leaves_.insert(k);
    }
  }
  
  void FindShallowLeaves(KeyType k, KeyType localroot, std::vector<KeyType> &set) {
    auto &h = data_->ht_;

    if (h.count(k) == 0) {
      return;
    }
    if (SFC::GetDepth(k) > SFC::GetDepth(localroot) + 2) {
      return;
    }

    if (h[k]->IsLeaf()) {
      set.push_back(k);
    } else {
      for (KeyType chk : SFC::GetChildren(k)) {
        FindShallowLeaves(chk, localroot, set);
      }
    }
  }



  /**
   * Exchange bodies and body attributes according to other processes according to proc_first_keys
   */
  std::tuple<std::vector<BodyType>, std::vector<BodyAttrType>>
             ExchangeBodies(std::vector<BodyType> bodies,
                            std::vector<BodyAttrType> attrs,
                            const std::vector<KeyType> proc_first_keys,
                            const Reg &reg, MPI_Comm comm) {
    std::vector<KeyType> body_keys = BodiesToKeys(bodies, reg);
    std::vector<int> dest(body_keys.size()); // destiantion of each body

    auto b = proc_first_keys.begin();
    auto e = proc_first_keys.end();

    for (size_t i = 0; i < body_keys.size(); i++) {
      dest[i] = std::upper_bound(b, e, body_keys[i]) - b - 1; // destination process
      TAPAS_ASSERT(0 <= dest[i]);
    }

    tapas::SortByKeys(dest, bodies, attrs);

    std::vector<BodyType> recv_bodies;
    std::vector<BodyAttrType> recv_attrs;
    std::vector<int> src;

    tapas::mpi::Alltoallv2(bodies, dest, recv_bodies, src, data_->mpi_type_body_, comm); // MPI_COMM_WORLD
    tapas::mpi::Alltoallv2(attrs,  dest, recv_attrs, src, data_->mpi_type_battr_, comm); // MPI_COMM_WORLD

#ifdef TAPAS_DEBUG // check
    // check if the total number of bodies

    // Check the given `bodies`
    int nb = bodies.size();
    int nb2 = attrs.size();
    int nb_total = 0;
    int nb_total2 = 0;
    tapas::mpi::Reduce(nb, nb_total, MPI_SUM, 0, comm);
    tapas::mpi::Reduce(nb2, nb_total2, MPI_SUM, 0, comm);

    if (data_->mpi_rank_ == 0) {
      assert(nb == nb2);
      assert(nb_total == (int)data_->nb_total);
      assert(nb_total2 == (int)data_->nb_total);
    }

    // Check the resulting `recv_bodies`
    nb = recv_bodies.size();
    nb2 = recv_attrs.size();
    nb_total = 0;
    nb_total2 = 0;
    tapas::mpi::Reduce(nb, nb_total, MPI_SUM, 0, comm);
    tapas::mpi::Reduce(nb2, nb_total2, MPI_SUM, 0, comm);

    if (data_->mpi_rank_ == 0) {
      assert(nb_total == (int)data_->nb_total);
      assert(nb_total2 == (int)data_->nb_total);
    }
#endif

    return std::make_tuple(recv_bodies, recv_attrs);
  }

  /**
   * \brief Returns the rank of the domain decomposition process (DD-process)
   * For now, the rank 0 process always does this.
   */
  static int DDProcId() {
    return 0;
  }

  template<class VecT>
  static std::vector<KeyType> BodiesToKeys(const VecT &bodies, const Reg &region) {
    return BodiesToKeys(bodies.begin(), bodies.end(), region);
  }

  /**
   * \brief Transform a vector of bodies into a vector of keys
   * \param[in] beg Iterator pointing the beginning of a vector of bodies
   * \param[in] end Iterator pointing the ned of a vector of bodies
   * \param[in] region Region of the global simulation space (returned by ExchangeRegion()).
   * \return           Vector of keys
   */
  template<class Iter>
  static std::vector<KeyType> BodiesToKeys(Iter beg, Iter end, const Reg &region) {
    unsigned long num_finest_cells = 1ul << SFC::MAX_DEPTH; // maximum number of cells in one dimension

    std::vector<KeyType> keys; // return value

    Vec<kDim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < kDim; ++d) {
      pitch[d] = (region.max()[d] - region.min()[d]) / (FP)num_finest_cells;
    }

    auto ins = std::back_inserter(keys);

    // For each bodies
    for (auto iter = beg; iter != end; iter++) {
      // Read the coordinates of the body (= *iter)
      // Coordinates are located after kPosOffset bytes from the head of the body structure.
      Vec<kDim, FP> pos = ParticlePosOffset<kDim, FP, kPosOffset>::vec(reinterpret_cast<const void*>(&*iter));
      Vec<kDim, FP> ofst = (pos - region.min()) / pitch;

      Vec<kDim, int> anchor; // An SFC key-like, but SoA-format vector without depth information. (Note that SFC keys are AoS format).

      // now ofst is a kDim-dimensional index of the finest-level cell to which the body belongs.
      for (int d = 0; d < kDim; d++) {
        anchor[d] = (int)ofst[d];
        
        if ((unsigned long)anchor[d] == num_finest_cells) {
          // the body is just on the upper edge so anchor[d] is over the
#ifdef TAPAS_DEBUG
          //TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
#endif
          anchor[d]--;
        }
      }

      *ins = SFC::CalcFinestKey(anchor);
    }

    return keys;
  }
};

} // namespace hot
} // namespace tapas


#endif // TAPAS_HOT_BUILDTREE_H
