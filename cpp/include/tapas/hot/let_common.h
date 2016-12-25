#ifndef TAPAS_HOT_LET_COMMON_
#define TAPAS_HOT_LET_COMMON_

#include <string>

namespace tapas {
namespace hot {

/**
 * Enum values of predicate function
 */
enum class InteractionType {
  Approx     = 1 << 0,   // Compute using right cell's attribute
  Body       = 1 << 1 ,  // Compute using right cell's bodies
  SplitLeft  = 1 << 2,   // Split left (local) cell
  SplitRight = 1 << 3,   // Split right (remote) cell
  SplitBoth  = SplitLeft | SplitRight,    // Split both cells
  SplitRightILL = 1 << 4,  // Split right leaf is split only *If the Left cell is a Leaf*
  None       = 0,          // Nothing. Use when a target cell isn't local in Traverse
};

std::string ToString(InteractionType st) {
  switch(st) {
    case InteractionType::Approx:        return "InteractionType::Approx";
    case InteractionType::Body:          return "InteractionType::Body";
    case InteractionType::SplitLeft:     return "InteractionType::SplitLeft";
    case InteractionType::SplitRight:    return "InteractionType::SplitRight";
    case InteractionType::SplitRightILL: return "InteractionType::SplitRightILL";
    case InteractionType::SplitBoth:     return "InteractionType::SplitBoth";
    case InteractionType::None:          return "InteractionType::None";
    default:
      assert(0);
      return "";
  }
}

std::ostream& operator<<(std::ostream &os, InteractionType st) {
  os << ToString(st);
  return os;
}

} /* namespace hot */
} /* namespace tapas */

#endif /* TAPAS_HOT_LET_COMMON_ */
