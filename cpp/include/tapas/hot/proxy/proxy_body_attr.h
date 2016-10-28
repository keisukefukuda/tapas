#ifndef TAPAS_HOT_PROXY_PROXY_BODY_ATTR_H_
#define TAPAS_HOT_PROXY_PROXY_BODY_ATTR_H_

namespace tapas {
namespace hot {
namespace proxy {

template<class PROXY_CELL>
class ProxyBodyAttr : public PROXY_CELL::TSP::BodyAttr {

  using TSP = typename PROXY_CELL::TSP;
  using BodyAttr = typename TSP::BodyAttr;
 public:
  ProxyBodyAttr(BodyAttr &rhs) : BodyAttr(rhs) {
  }

  template <class T>
  inline ProxyBodyAttr& operator=(const T &) {
    return *this;
  }

  template<class T>
  inline const ProxyBodyAttr& operator=(const T &) const {
    return *this;
  }
}; // class ProxyBodyAttr

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_PROXY_BODY_ATTR_H_
