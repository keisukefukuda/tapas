#ifndef TAPAS_HOT_INSP1_H_
#define TAPAS_HOT_INSP1_H_

#include <tapas/hot/proxy/proxy_cell.h>

#include <tapas/hot/proxy/full_traverse_policy.h>

namespace tapas {
namespace hot {

template<class TSP> class Cell;

/**
 * \brief Inspector implementation for Map-1
 */ 
template<class TSP>
struct Insp1 {
  // type aliases
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using CellType = Cell<TSP>;
  using Key = typename CellType::KeyType;
  using SFC = typename CellType::SFC;
  using Data = typename CellType::Data;
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  using Body = typename CellType::BodyType;
  using BodyAttr = typename CellType::BodyAttrType;
  
  using CellAttr = typename CellType::CellAttr;
  using Vec = tapas::Vec<TSP::Dim, typename TSP::FP>;
  using Reg = Region<Dim, FP>;

  using ProxyCell = tapas::hot::proxy::ProxyCell<TSP, tapas::hot::proxy::FullTraversePolicy<TSP>>;

  /**
   * Direction of Map-1 (Upward/Downward)
   */
  enum {
    MAP1_UP,
    MAP1_DOWN,
    MAP1_UNKNOWN
  };

  /**
   * \brief Inspect user function for 1-map and returns MAP1_UP, MAP1_DOWN, or MAP1_UNKNOWN
   */ 
  template<class Funct, class...Args>
  static int FindMap1Direction(CellType &c, Funct f, Args...args) {
    const Key kRoot = 0;

    return PredDir1(kRoot, c.data(), f, args...);
  }

  // Determine direction (upward or downward) of Map-1
  template<class UserFunct, class...Args>
  static int PredDir1(Key key, const Data &data, UserFunct f, Args...args) {
    int clock = 1;
    ProxyCell parent(data, &clock, key);

    ProxyCell &child = parent.subcell(0);

    f(parent, child, args...);

    // if cell is modified
    int lv0_mod = parent.IsMarkedModified();
      
    // if any of the children is modified
    int lv1_mod = child.IsMarkedModified();

    // lv0_mod and lv1_mod represents the timing of modification to the cell.
    // here `time' is measured by the 'clock' value.
      
    if (lv0_mod > lv1_mod) { // level 0 cell was updated later => Upward
      // Upward
      return MAP1_UP;
    } else if (lv0_mod < lv1_mod) { // level 1 cell was updated later => Downward
      // Downward
      return MAP1_DOWN;
    } else {
      // lv0_mod == lv1_mod
      // Leaf-only traversal or non-destructive traverse (such as debug printing)
      return MAP1_UNKNOWN;
    }
  }
};

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_INSP1_H_


