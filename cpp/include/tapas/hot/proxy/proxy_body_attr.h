#ifndef TAPAS_HOT_PROXY_PROXY_BODY_ATTR_H_
#define TAPAS_HOT_PROXY_PROXY_BODY_ATTR_H_

namespace tapas {
namespace hot {
namespace proxy {

template<class BODY, class BODY_ATTR>
class ProxyBodyAttr : public BODY_ATTR {
  using BodyAttr = BODY_ATTR;
  using Body = BODY;
 public:
  ProxyBodyAttr(BodyAttr &rhs) : BodyAttr(rhs) { }
  ProxyBodyAttr() : BodyAttr() { }

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
