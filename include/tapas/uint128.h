#ifndef TAPAS_UINT128_H__
#define TAPAS_UINT128_H__

#include <ostream>

// Do all compilers have these extensions?
#if defined(TAPAS_COMPILER_GCC)
using int128_t = __int128_t;
using uint128_t = __uint128_t;
#elfi defined(TAPAS_COMPILER_CLANG)
using int128_t = __int128_t;
using uint128_t = __uint128_t;
#elif defined(TAPAS_COMPILER_INTEL)
using int128_t = __int128_t;
using uint128_t = __uint128_t;
#elif defined(__CUDACC__)
using int128_t = __int128;
using uint128_t = unsigned __int128;
#else
using int128_t = __int128_t;
using uint128_t = unsigned __int128_t;
#endif

namespace std {

// hash function for std::unordered_map.
template<>
struct hash<uint128_t> {
  std::size_t operator()(const uint128_t &k) const {
    using std::hash;
    uint64_t upper = (uint64_t)(k >> 64);
    uint64_t lower = (uint64_t)(k & 0xffffffff);
    return (size_t)(lower ^ upper);
  }
};

std::ostream& operator<<(std::ostream& dest, uint128_t value) {
  const char *nums = "0123456789";
  std::ostream::sentry s(dest);
  if (s) {
    char buffer[64];
    char* e = std::end(buffer);
    do {
      --e;
      *e = nums[value % 10];
      value /= 10;
    } while ( value != 0 );
    dest << e;
  }
  return dest;
}

} // namespace std

#endif // TAPAS_UINT128_H__
