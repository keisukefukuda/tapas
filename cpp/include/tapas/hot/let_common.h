#ifndef TAPAS_HOT_LET_COMMON_
#define TAPAS_HOT_LET_COMMON_

#include <string>

namespace tapas {
namespace hot {

// Interaction Flag class
class IntrFlag {
  using type = uint32_t;
 public:
  static constexpr type ReadAttrR = 1 << 0; // Cell::attr() of right cell is called
  static constexpr type ReadAttrL = 1 << 1; // Cell::attr() of left cell is called
  static constexpr type ReadAttr  = ReadAttrL | ReadAttrR;
  static constexpr type ReadNbR   = 1 << 2; // Cell::nb() of right cell is called
  static constexpr type ReadNbL   = 1 << 3; // Cell::nb() of left cell is called
  static constexpr type ReadNb    = ReadNbR | ReadNbL;
  static constexpr type SplitR    = 1 << 4; // the right cell is split
  static constexpr type SplitL    = 1 << 5; // the left cell is split
  static constexpr type Split     = SplitR | SplitL;
  static constexpr type SplitILL  = 1 << 6; // Split the right cell only if the left cell is a leaf.

  IntrFlag(type flag) : flag_(flag) {}
  IntrFlag() : flag_(0) {}

  void Add(type f) {
    flag_ |= f;
  }

  IntrFlag operator|(const IntrFlag &rhs) {
    return IntrFlag(flag_ | rhs.flag_);
  }

  bool operator==(const IntrFlag &rhs) {
    return flag_ == rhs.flag_;
  }
  
  bool IsApprox() const {
    return !( (flag_ & SplitR) ||
              (flag_ & SplitL) ||
              (flag_ & SplitILL) );
  }

  inline bool IsSplitR() const { return flag_ & SplitR; }
  inline bool IsSplitL() const { return flag_ & SplitL; }
  inline bool IsSplitILL() const { return flag_ & SplitILL; }
  inline bool IsSplit() const { return IsSplitR() || IsSplitL() || IsSplitILL(); }
        
  inline bool IsReadAttrR() const { return flag_ & ReadAttrR; }
  inline bool IsReadAttrL() const { return flag_ & ReadAttrL; }
  inline bool IsReadAttr() const { return IsReadAttrR() || IsReadAttrL(); }
  
  std::string ToString() const {
    std::vector<std::string> res;

    if (flag_ & ReadAttrR) res.push_back("ReadAttrR");
    if (flag_ & ReadAttrL) res.push_back("ReadAttrL");
    if (flag_ & ReadNbR)   res.push_back("ReadNbR");
    if (flag_ & ReadNbL)   res.push_back("ReadNbL");
    if (flag_ & SplitR)    res.push_back("SplitR");
    if (flag_ & SplitL)    res.push_back("SplitL");
    if (flag_ & SplitILL)    res.push_back("SplitILL");

    std::string out;
    for (size_t i = 0; i < res.size(); i++) {
      if (i > 0) out += " | ";
      out += res[i];
    }

    return out;
  }

 private:
  type flag_;
};

} /* namespace hot */
} /* namespace tapas */

#endif /* TAPAS_HOT_LET_COMMON_ */
