#ifndef TAPAS_HOT_DATA_H_
#define TAPAS_HOT_DATA_H_

#include <unordered_map>
#include <string>
#include <vector>

namespace tapas {
namespace hot {

// fwd decl
template<class TSP> class Cell;
template<class TSP> class DummyCell;

/**
 * \brief Struct to hold shared data among Cells
 */
template<class TSP, class SFC_>
struct SharedData {
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using SFC = SFC_;
  using KeyType = typename SFC::KeyType;
  using Reg = Region<Dim, FP>;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using KeySet = std::unordered_set<KeyType>;
  using CellAttr = typename TSP::CellAttr;
  using BodyType = typename TSP::Body;
  using BodyAttrType = typename TSP::BodyAttr;
  using Mapper = typename CellType::Mapper;

  // To be removed
  //template<class T> using Allocator = typename TSP::template Allocator<T>;

  CellHashTable ht_;
  CellHashTable ht_let_;

  CellHashTable ht_gtree_; // Hsah table of the global tree.
  KeySet        gleaves_;  // set of global leaves, which are a part of ht_gtree_.keys and ht_.keys
  KeySet        lroots_;   // set of local roots. It must be a subset of gleaves_. gleaves_ is "Allgatherv-ed" lroots.
  std::mutex    ht_mtx_;   //!< mutex to protect ht_
  KeySet        let_used_key_;
  std::unordered_map<KeyType, CellAttr> local_upw_results_; // used in Upward Map

  std::vector<Reg> local_br_; // Bounding Region of the local process. local_br_ = [ C.region() for C in lroots_ ]
  // std::set might be better, but Region needs to have operator<() for that.

  Reg region_;   //!< global bouding box

  Mapper mapper_;

  // debug LET comm
  //KeySet        trav_used_src_key_;

  int mpi_rank_;
  int mpi_size_;
  MPI_Comm mpi_comm_;
  int max_depth_; //!< Actual maximum depth of the tree

  // removed
  //Vec<Dim, FP> local_bb_max_; //!< Coordinates of the bounding box of the local process
  //Vec<Dim, FP> local_bb_min_; //!< Coordinates of the bounding box of the local process

  std::vector<KeyType> leaf_keys_; //!< SFC keys of (all) leaves
  std::vector<index_t> leaf_nb_;   //!< Number of bodies in each leaf cell
  std::vector<int>     leaf_owners_; //!< Owner process of leaf[i]

#if 0
  std::vector<BodyType, Allocator<BodyType>> local_bodies_; //!< Bodies that belong to the local process
  std::vector<BodyType, Allocator<BodyType>> let_bodies_; //!< Bodies sent from remote processes
  std::vector<BodyAttrType, Allocator<BodyAttrType>> local_body_attrs_; //!< Local body attributes
  std::vector<BodyAttrType, Allocator<BodyAttrType>> let_body_attrs_; //!< Local body attributes
#else
  std::vector<BodyType> local_bodies_; //!< Bodies that belong to the local process
  std::vector<BodyType> let_bodies_; //!< Bodies sent from remote processes
  std::vector<BodyAttrType> local_body_attrs_; //!< Local body attributes
  std::vector<BodyAttrType> let_body_attrs_; //!< Local body attributes
#endif

  std::vector<KeyType>  local_body_keys_; //!< SFC keys of local bodies

  std::vector<KeyType> proc_first_keys_; //!< first SFC key of each process

  int opt_task_spawn_threshold_;

  // log and time measurements (mainly of the local process)
  double sampling_rate; // sampling rate of tree construction
  index_t nb_total;  // total number of bodies.
  index_t nb_before; // local bodies before tree construction (given by the user)
  index_t nb_after;  // local bodies after tree construction  (actuall)
  index_t nleaves;   // number of leaves assigned to the local process
  index_t ncells;    // number of cells (note: some non-leaf cells are shared between processes)

  int count_map1_; // How many times 2-parameter Map() is called so far within the current timestep
  int count_map2_; // How many times 2-parameter Map() is called so far within the current timestep
  tapas::util::TimeRec time_rec_;

  std::unordered_map<int, int> let_func_count;

  MPI_Datatype mpi_type_key_;
  MPI_Datatype mpi_type_attr_;
  MPI_Datatype mpi_type_body_;
  MPI_Datatype mpi_type_battr_;

  SharedData()
      : mpi_rank_(0)
      , mpi_size_(1)
      , mpi_comm_(MPI_COMM_WORLD)
      , max_depth_(0)
      , opt_task_spawn_threshold_(1000)
      , nb_total(0)
      , nb_before(0)
      , nb_after(0)
      , nleaves(0)
      , ncells(0)
      , count_map1_(0)
      , count_map2_(0)
      , time_rec_()
      , let_func_count()
  {
    ReadEnv();
    SetupMPITypes();
  }

  SharedData(const SharedData<TSP, SFC>& rhs) = delete; // no copy
  SharedData(SharedData<TSP, SFC>&& rhs) = delete; // no move

  /**
   * \brief Read some of the parameters from environmental variables
   */
  void ReadEnv() {
    const char *var = nullptr;
    if ((var = getenv("TAPAS_TASK_SPAWN_THRESHOLD"))) {
      opt_task_spawn_threshold_ = atoi(var);
      if (opt_task_spawn_threshold_ <= 0) {
        std::cout << "TAPAS_TASK_SPAWN_THRESHOLD must be > 0" << std::endl;
        exit(-1);
      }
    }
  }

  void SetupMPITypes() {
    // Cell Attributes
    MPI_Type_contiguous(sizeof(KeyType), MPI_BYTE, &mpi_type_key_);
    MPI_Type_contiguous(sizeof(CellAttr), MPI_BYTE, &mpi_type_attr_);
    MPI_Type_contiguous(sizeof(BodyType), MPI_BYTE, &mpi_type_body_);
    MPI_Type_contiguous(sizeof(BodyAttrType), MPI_BYTE, &mpi_type_battr_);

    MPI_Type_commit(&mpi_type_key_);
    MPI_Type_commit(&mpi_type_attr_);
    MPI_Type_commit(&mpi_type_body_);
    MPI_Type_commit(&mpi_type_battr_);
  }
};


} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_DATA_H_
