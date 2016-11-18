#ifndef TAPAS_HOT_GLOBAL_TREE_H
#define TAPAS_HOT_GLOBAL_TREE_H

#include "tapas/stdcbug.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

#include <mpi.h>

#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/mpi_util.h"

#include "tapas/hot/shared_data.h"

namespace tapas {
namespace hot {

template<typename TSP> class Cell;

template<class TSP>
class GlobalTree {
 public:

  using CellType = Cell<TSP>;
  using Reg = typename CellType::Reg;
  using SFC = typename CellType::SFC;
  using Data = SharedData<TSP, SFC>;
  using KeyType = typename CellType::KeyType;
  using HashTable = typename CellType::CellHashTable;
  using KeySet = typename CellType::KeySet;
  using FP = typename TSP::FP;
  static const constexpr int Dim = TSP::Dim;

  /**
   * \brief Build global tree
   * 1. Traverse recursively from the root and identify global leaves
   * 2. Exchange global leaves using Alltoallv
   * 3. Build global tree locally
   *
   * Called from Partition() in hot.h
   */
  static void Build(Data &data) {
    double beg = MPI_Wtime();

    HashTable &ltree = data.ht_;       // hash table for the local tree
    HashTable &gtree = data.ht_gtree_; // hash table for the global tree
    KeySet &gleaves = data.gleaves_;
    KeySet &lroots = data.lroots_;

    gtree.clear();

    FindLocalRoots(0, ltree, lroots);

    // construct minimum depth map for each local roots
    // (used in one-side LET inspector)
    // for (KeyType lr : lroots) {
    //   FindMinLeafLevel(lr, lr, data.ht_, data.min_leaf_level_);
    //   std::cout << "min leaf level: " << SFC::Decode(lr) << " : " << data.min_leaf_level_[lr] << std::endl;
    // }

    // Exchange global leaves using Allgatherv
    ExchangeGlobalLeafKeys(lroots, gleaves);

    // Glow the global tree locally in each process
    GrowGlobalTree(gleaves, data.ht_, data.ht_gtree_);

    GetLocalBB(data);

    double end = MPI_Wtime();
    data.time_rec_.Record(data.timestep_, "Tree-growglobal", end - beg);
  }

  /**
   * \brief Calculate local bouding box
   * Used in one-sided (conservative, approximate) LET traversal
   */
  static void GetLocalBB(Data &data) {
    using T = decltype(data.local_br_);

    std::transform(data.lroots_.begin(),
                   data.lroots_.end(),
                   std::insert_iterator<T>(data.local_br_, data.local_br_.begin()),
                   [&data](KeyType k) {
                     return data.ht_[k]->region();
                   });
  }

  /**
   * \brief A reducing function used with LocalUpwardReduce, to check if a cell is a local subtree.
   */
  static bool ReduceLocality(bool b, KeyType, const CellType *c) {
    return b && c != nullptr && c->IsLocalSubtree();
  }

  static void FindMinLeafLevel(KeyType local_root, KeyType key,
                               const std::unordered_map<KeyType, CellType*>& ht,
                               std::unordered_map<KeyType, int>& min_leaf_level) {
    if (ht.count(key) > 0) {
      if (ht.at(key)->IsLeaf()) {
        int cur_min_dep = min_leaf_level.count(local_root) > 0 ? min_leaf_level[local_root] : 99999;
        min_leaf_level[local_root] = std::min(cur_min_dep, SFC::GetDepth(key));
      } else {
        for (KeyType ch : SFC::GetChildren(key)) {
          FindMinLeafLevel(local_root, ch, ht, min_leaf_level);
        }
      }
    }
  }
  
  static void FindLocalRoots(KeyType key, const HashTable &ht, KeySet &lroots) {
    TAPAS_ASSERT(ht.count(key) == 1);

    CellType *c = ht.at(key);

    LocalUpwardReduce(c, &CellType::is_local_subtree_, true,
                      &GlobalTree<TSP>::ReduceLocality);

    // Create a closure to find all local roots.
    auto local_root_collector = [&lroots] (const CellType *c) -> bool {
      if (c->is_local_subtree_) {
        lroots.insert(c->key());
        return false;
      } else {
        return true;
      }
    };

    // Find all local subtree roots.
    LocalPreOrderTraverse(c, local_root_collector);
  }


  /**
   * \brief Performs pre-order traversal for local cells.
   * \tparam TSP TSP.
   * \tparam Funct A callback function called for each cell. Expected to be <bool (CellType*)>.
   * \param c A cell to start traversal
   * \param f A callback function. If f returns false for a cell, then its children are not visited.
   */
  template<class Funct>
  static void LocalPreOrderTraverse(CellType *c, Funct f) {
    TAPAS_ASSERT(c->IsLocal());

    auto &data = c->data();

    bool cont = f(c);
    if (!cont) return;

    if (!c->IsLeaf()) {
      auto child_keys = SFC::GetChildren(c->key());
      for (auto chk : child_keys) {
        if (data.ht_.count(chk) > 0) {
          Cell<TSP> *cc = data.ht_.at(chk);
          LocalPreOrderTraverse(cc, f);
        }
      }
    }
  }

  /**
   * \brief A functional utility that provides upward reudce of a local tree for internal use.
   *
   * Non-local cells are just ignored.
   * Values of member variables (memvp) of child cells are reduce using function f and
   * assigned into parent's memvp value.
   * If c is a leaf, c.*memvp is set to `init`
   *
   * \tparam T    Type of target member values.
   * \tparam Funct Type of the reducing function f. It is expected to be T(T, KeType, const CellType*c).
   * \param c     Starting root cell
   * \param memvp A member variable pointer (obtained by &Class::member)
   * \param init  Initial value
   * \param f     A reducing function of type (T, KeyType, const Cell*) -> T. If a child cell is local,
   *              KeyType and Cell* are both given. If a child is not local, the pointer is nullptr.
   */
  template<class T, class Funct>
  static void LocalUpwardReduce(CellType *c, T CellType::*memvp, T init, Funct f) {
    if (c->IsLeaf()) {
      // If c is a leaf, just assign the value `init` to `memvp` member variable.
      c->*memvp = init;
    } else {
      // c is a non-leaf cell. Reduce children's value and assign it into c->*memvp
      auto &data = c->data();
      KeyType key = c->key();
      auto children_keys = SFC::GetChildren(key);

      T val = init;

      for (auto chk : children_keys) {
        if (data.ht_.count(chk) > 0) {
          Cell<TSP> *cc = data.ht_.at(chk);
          LocalUpwardReduce(cc, memvp, init, f);
          val = f(val, chk, const_cast<const CellType*>(cc));
        } else {
          val = f(val, chk, nullptr);
        }
      }

      c->*memvp = val; // assign reduced value.
    }
  }

  static void ExchangeGlobalLeafKeys(const KeySet &lroots, KeySet &gleaves) {
    std::vector<KeyType> gl_keys_send(lroots.begin(), lroots.end()); // global leaf keys
    std::vector<KeyType> gl_keys_recv;
    tapas::mpi::Allgatherv(gl_keys_send, gl_keys_recv, MPI_COMM_WORLD);

    gleaves.insert(gl_keys_recv.begin(), gl_keys_recv.end());
  }

  static void GrowGlobalTree(const KeySet &gleaves, const HashTable &ht, HashTable &ht_gtree) {
    Data *dp = ht.at(0)->data_ptr();

    for (auto && key : gleaves) {
      for (KeyType k = key; k != 0; k = SFC::Parent(k)) {
        if (ht_gtree.count(k) > 0) {
          break; // break to outer loop
        } else if (ht.count(k) > 0) {
          ht_gtree[k] = ht.at(k);
        } else {
          ht_gtree[k] = CellType::CreateRemoteCell(k, 0, dp);
        }
      }
    }
    ht_gtree[0] = ht.at(0);
  }
};

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_GLOBAL_TREE_H
