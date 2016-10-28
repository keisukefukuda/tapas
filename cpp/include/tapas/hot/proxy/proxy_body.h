#ifndef TAPAS_PROXY_BODY_H_
#define TAPAS_PROXY_BODY_H_

/**
 * \brief Proxy class for user-specified Body class
 */
template<class PROXY_CELL>
class ProxyBody : public PROXY_CELL::TSP::Body {
  using Body = typename PROXY_CELL::TSP::Body;
  using BodyAttr = typename PROXY_CELL::TSP::BodyAttr;
  
 public:
  ProxyBody(BodyAttr &rhs) : Body(rhs) {
  }
};

#endif // TAPAS_PROXY_BODY_H_
