#ifndef TAPAS_HOT_PROXY_PROXY_ATTR_H_
#define TAPAS_HOT_PROXY_PROXY_ATTR_H_

namespace tapas {
namespace hot {
namespace proxy {

/**
 * \brief Proxy class for user-specified CellAttr.
 */
template<class PROXY_CELL>
class ProxyAttr : public PROXY_CELL::TSP::CellAttr {
  friend PROXY_CELL;
  using CellAttr = typename PROXY_CELL::TSP::CellAttr; // real cell attributes
    
 protected:
  ProxyAttr &operator=(const ProxyAttr &rhs) {
    this->CellAttr::operator=(rhs);
    cell_ = rhs.cell_;
    return *this;
  }
    
 public:
  ProxyAttr(PROXY_CELL *cell) : CellAttr(), cell_(cell) { }
  ProxyAttr(PROXY_CELL *cell, CellAttr &rhs) : CellAttr(rhs), cell_(cell) { }

  //ProxyAttr(const ProxyAttr &) = delete;
    
  PROXY_CELL &cell() const { return *cell_; }

  inline void operator=(const CellAttr&) const {
    //std::cout << "ProxyAttr::operator=() is called for cell [" << cell_->key() << "]" << std::endl;
    cell_->MarkModified();
    //return *this;
  }

  template<class T>
  inline void operator=(const CellAttr&) const {
    //std::cout << "ProxyAttr::operator=() const is called for cell [" << cell_->key() << "]" << std::endl;
    cell_->MarkModified();
    //return *this;
  }

 private:
  PROXY_CELL *cell_;
}; // class ProxyAttr


} // namespace proxy
} // namespace hot
} // namespace proxy

#endif // TAPAS_HOT_PROXY_PROXY_ATTR_H_
