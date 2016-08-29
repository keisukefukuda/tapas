#ifndef TAPAS_PROXY_BODY_H_
#define TAPAS_PROXY_BODY_H_

/**
 * \brief Proxy class for user-specified Body class
 */
template<class TSP>
class ProxyBody : public TSP::Body {
  using Body = typename TSP::Body;
  using BodyAttr = typename TSP::BodyAttr;
  
 public:
  ProxyBody(BodyAttr &rhs) : Body(rhs) {
  }
};

#endif // TAPAS_PROXY_BODY_H_
