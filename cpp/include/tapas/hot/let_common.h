#ifndef TAPAS_HOT_LET_COMMON_
#define TAPAS_HOT_LET_COMMON_

#include <string>

namespace tapas {
namespace hot {

/**
 * Enum values of predicate function
 */
enum class SplitType {
  Approx     = 1 << 0,   // Compute using right cell's attribute
  Body       = 1 << 1 ,  // Compute using right cell's bodies
  SplitLeft  = 1 << 2,   // Split left (local) cell
  SplitRight = 1 << 3,   // Split right (remote) cell
  SplitRightILL,         // Split right leaf is split only *If the Left cell is a Leaf*
  SplitBoth  = SplitLeft | SplitRight,    // Split both cells
  None       = 0,        // Nothing. Use when a target cell isn't local in Traverse
};

std::string ToString(SplitType st) {
  switch(st) {
    case SplitType::Approx:        return "SplitType::Approx";
    case SplitType::Body:          return "SplitType::Body";
    case SplitType::SplitLeft:     return "SplitType::SplitLeft";
    case SplitType::SplitRight:    return "SplitType::SplitRight";
    case SplitType::SplitRightILL: return "SplitType::SplitRightILL";
    case SplitType::SplitBoth:     return "SplitType::SplitBoth";
    case SplitType::None:          return "SplitType::None";
    default:
      assert(0);
      return "";
  }
}

std::ostream& operator<<(std::ostream &os, SplitType st) {
  os << ToString(st);
  return os;
}

} /* namespace hot */
} /* namespace tapas */

#endif /* TAPAS_HOT_LET_COMMON_ */
