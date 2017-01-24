#ifndef TAPAS_PROXY_BODY_H_
#define TAPAS_PROXY_BODY_H_

namespace tapas {
namespace hot {
namespace proxy {

/**
 * \brief Proxy class for user-specified Body class
 */
template<class BODY_TYPE, class BODY_ATTR_TYPE, class TRAVERSE_POLICY>
class ProxyBody : public BODY_TYPE {
  using Body = BODY_TYPE;
  using BodyAttr = BODY_ATTR_TYPE;
  using TraversePolicy = TRAVERSE_POLICY;

 public:
  ProxyBody(BodyAttr &rhs) : Body(rhs), parent_(nullptr) { }

  ProxyBody() : Body(), parent_(nullptr) { }

  void SetParent(TraversePolicy *p) {
    parent_ = p;
  }

  const TraversePolicy *Parent() const {
    return parent_;
  }

  TraversePolicy *Parent() {
    return parent_;
  }

 private:
  TraversePolicy *parent_;
};

}}}

#endif // TAPAS_PROXY_BODY_H_
