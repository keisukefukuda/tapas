#ifndef TAPAS_PROXY_BODY_H_
#define TAPAS_PROXY_BODY_H_

/**
 * \brief Proxy class for user-specified Body class
 */
template<class BODY_TYPE, class BODY_ATTR_TYPE>
class ProxyBody : public BODY_TYPE {
  using Body = BODY_TYPE;
  using BodyAttr = BODY_ATTR_TYPE;
  
 public:
  ProxyBody(BodyAttr &rhs) : Body(rhs) { }

  ProxyBody() : Body() { }
};

#endif // TAPAS_PROXY_BODY_H_
