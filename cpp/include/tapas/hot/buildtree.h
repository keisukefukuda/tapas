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
  std::vector<double> weights_;  // weights of bodies
  std::vector<KeyType> body_keys_;
  std::vector<KeyType> proc_first_keys_; // first key of each process's region
  Reg region_;
  Data* data_;
  int ncrit_;

 public:
  SamplingOctree(const BodyType *b, const double *w, index_t nb, Data *data, int ncrit)
      : bodies_(b, b+nb)
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

  static void ShowHistogram(const Data &data) {
    using tapas::debug::BarrierExec;
    const int d = data.max_depth_;
    TAPAS_ASSERT(d <= SFC::MaxDepth());

    // Show "level-#cells" histogram

    const long ncells = data.ht_.size();
    const long nall   = (pow(8.0, d+1) - 1) / 7;
    BarrierExec([&](int,int) {
        std::cout << "Cells: " << ncells << std::endl;
        std::cout << "depth: " << d << std::endl;
        std::cout << "filling rate: " << ((double)ncells / nall) << std::endl;
      });

    std::vector<int> hist(d + 1, 0);
    for (auto p : data.ht_) {
      const auto *cell = p.second;
      if (cell->IsLeaf()) {
        hist[cell->depth()]++;
      }
    }

    BarrierExec([&](int, int) {
        std::cout << "Depth histogram" << std::endl;
        for (int i = 0; i <= d; i++) {
          std::cout << i << " " << hist[i] << std::endl;
        }
      });

    // Show "Region-#cells" histogram to show how biased the distribution is.

    int bits = SFC::Dim * 2;
    int num_classes = 1 << bits;
    std::vector<int> count(num_classes);
    for (auto kk : data.ht_) {
      KeyType key = kk.first;
      KeyType n = SFC::GetMSBits(key, bits);
      count[n]++;
    }

    std::vector<int> recv_buf(num_classes);
    MPI_Reduce(&count[0], &recv_buf[0], count.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (data.mpi_rank_ == 0) {
      std::cout << "Total cells: " << data.ht_.size() << std::endl;
      std::cout << "Region Histogram: ";
      for(int i = 0; i < num_classes; i++) {
        std::cout << recv_buf[i] << " ";
      }
      std::cout << std::endl;
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
    
    const int B = 1 << kDim; // 8 in 3-dim space
    const int Ls  = (int)(log((double)mpi_size) / log((double)B) + 2); // logB(Np) = log(Np) / log(B)

    // total weight
    const double totalw = std::accumulate(body_weights.begin(), body_weights.end(), 0);

    // Beginning key of each process.
    // This is the target value of this function to be returned to the caller.
    std::vector<KeyType> beg_keys(mpi_size);
    proc_weights.resize(mpi_size);

    for (int L = Ls; L < std::min(Ls + 5, SFC::MaxDepth()); L++) {
      // Loop over [Ls, Ls+1, ...] until the load balancing seems good.
      
      // The value 'Ls + 5' is hardcoded.
      // If the value is too large, old-inspector takes longer
      // (because the hypothetical global tree gets higher)
      
      const KeyType K0 = SFC::AppendDepth(0, L); // the first key in level L
      const int W = pow(B, L); // number of cells in level L

      std::cout << "L=" << L << std::endl;
      if (W <= mpi_size) {
        std::cout << "W=" << W << ", mpi_size=" << mpi_size << std::endl;
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
        //std::cout << "Finding range of [b, e) for P=" << pi << " ... ";
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
        //std::cout << "pi=" << (pi-1) << " actual weight=" << proc_weights[pi-1] << std::endl;
        
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
      
#endif
      std::cout << "Ratio = " << ratio << std::endl;

      if (ratio < 0.01) break;
    }

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

#ifdef TAPAS_DEBUG
    // Output proc_weights for debugging/profiling purpose
    double pw; // proc's weight
    tapas::mpi::Scatter(proc_weights, pw, 0, data_->mpi_comm_);
    data_->time_rec_.Record(data_->timestep_, "Weight", pw);
#endif
  }

  /**
   * \brief Exchange bodies to owner processes determined by Sample() function.
   */
  void Exchange() {
    double beg = MPI_Wtime();

    data_->nb_before = bodies_.size();

    // Exchange bodies according to proc_first_keys_
    // new_bodies is the received bodies

    bodies_ = ExchangeBodies(bodies_, proc_first_keys_, region_, MPI_COMM_WORLD);
    body_keys_ = BodiesToKeys(bodies_, region_);

    // Sort both new_keys and new_bodies.
    SortByKeys(body_keys_, bodies_);

    // todo: record bodies

    data_->local_bodies_.assign(bodies_.begin(), bodies_.end());
    data_->local_body_keys_ = body_keys_;

    data_->local_body_attrs_.resize(bodies_.size());
    bzero(data_->local_body_attrs_.data(), sizeof(BodyAttrType) * bodies_.size());
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

#ifdef TAPAS_DEBUG_HISTOGRAM
    ShowHistogram(*data_);
#endif
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
      // The cell [k] is not a leaf. Split it again.
      // Note: if the cell is not a leaf and nb == 0, that means other processes may have particles which belong to the cell.
      auto ch_keys = SFC::GetChildren(k);

      for (auto chk : ch_keys) {
        // Check if the child key is in the range of this process, ignore it otherwise.
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

  std::vector<BodyType> ExchangeBodies(std::vector<BodyType> bodies,
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

    tapas::SortByKeys(dest, bodies);

    std::vector<BodyType> recv_bodies;
    std::vector<int> src;

    tapas::mpi::Alltoallv2(bodies, dest, recv_bodies, src, data_->mpi_type_body_, comm); // MPI_COMM_WORLD

#if 1 // debug
    // check if the total number of bodies

    // Check the given `bodies`
    int nb = bodies.size();
    int nb_total = 0;
    tapas::mpi::Reduce(nb, nb_total, MPI_SUM, 0, comm);

    if (data_->mpi_rank_ == 0) {
      assert(nb_total == (int)data_->nb_total);
    }

    // Check the resulting `recv_bodies`
    nb = recv_bodies.size();
    nb_total = 0;
    tapas::mpi::Reduce(nb, nb_total, MPI_SUM, 0, comm);

    if (data_->mpi_rank_ == 0) {
      assert(nb_total == (int)data_->nb_total);
    }
#endif

    return recv_bodies;
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
    int num_finest_cells = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension

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
        
        if (anchor[d] == num_finest_cells) {
          // the body is just on the upper edge so anchor[d] is over the
          TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
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
